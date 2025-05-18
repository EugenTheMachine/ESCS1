# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, List, Tuple, Type, Dict

import torch
import torch.nn.functional as F

from torch import nn, Tensor

from efficient_sam.efficient_sam_decoder import MaskDecoder, PromptEncoder
from efficient_sam.efficient_sam_encoder import ImageEncoderViT
from efficient_sam.two_way_transformer import TwoWayAttentionBlock, TwoWayTransformer

class EfficientSam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        decoder_max_num_input_points: int,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [0.485, 0.456, 0.406],
        pixel_std: List[float] = [0.229, 0.224, 0.225],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.decoder_max_num_input_points = decoder_max_num_input_points
        self.mask_decoder = mask_decoder
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(1, 3, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(pixel_std).view(1, 3, 1, 1), False
        )

    @torch.jit.export
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        batched_points: torch.Tensor,
        batched_point_labels: torch.Tensor,
        multimask_output: bool,
        input_h: int,
        input_w: int,
        output_h: int = -1,
        output_w: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts masks given image embeddings and prompts. This only runs the decoder.

        Arguments:
          image_embeddings: A tensor of shape [B, C, H, W] or [B*max_num_queries, C, H, W]
          batched_points: A tensor of shape [B, max_num_queries, num_pts, 2]
          batched_point_labels: A tensor of shape [B, max_num_queries, num_pts]
        Returns:
          A tuple of two tensors:
            low_res_mask: A tensor of shape [B, max_num_queries, 256, 256] of predicted masks
            iou_predictions: A tensor of shape [B, max_num_queries] of estimated IOU scores
        """

        batch_size, max_num_queries, num_pts, _ = batched_points.shape
        num_pts = batched_points.shape[2]
        rescaled_batched_points = self.get_rescaled_pts(batched_points, input_h, input_w)

        if num_pts > self.decoder_max_num_input_points:
            rescaled_batched_points = rescaled_batched_points[
                :, :, : self.decoder_max_num_input_points, :
            ]
            batched_point_labels = batched_point_labels[
                :, :, : self.decoder_max_num_input_points
            ]
        elif num_pts < self.decoder_max_num_input_points:
            rescaled_batched_points = F.pad(
                rescaled_batched_points,
                (0, 0, 0, self.decoder_max_num_input_points - num_pts),
                value=-1.0,
            )
            batched_point_labels = F.pad(
                batched_point_labels,
                (0, self.decoder_max_num_input_points - num_pts),
                value=-1.0,
            )

        sparse_embeddings = self.prompt_encoder(
            rescaled_batched_points.reshape(
                batch_size * max_num_queries, self.decoder_max_num_input_points, 2
            ),
            batched_point_labels.reshape(
                batch_size * max_num_queries, self.decoder_max_num_input_points
            ),
        )
        sparse_embeddings = sparse_embeddings.view(
            batch_size,
            max_num_queries,
            sparse_embeddings.shape[1],
            sparse_embeddings.shape[2],
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings,
            self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            multimask_output=multimask_output,
        )
        _, num_predictions, low_res_size, _ = low_res_masks.shape

        if output_w > 0 and output_h > 0:
            output_masks = F.interpolate(
                low_res_masks, (output_h, output_w), mode="bicubic"
            )
            output_masks = torch.reshape(
                output_masks,
                (batch_size, max_num_queries, num_predictions, output_h, output_w),
            )
        else:
            output_masks = torch.reshape(
                low_res_masks,
                (
                    batch_size,
                    max_num_queries,
                    num_predictions,
                    low_res_size,
                    low_res_size,
                ),
            )
        iou_predictions = torch.reshape(
            iou_predictions, (batch_size, max_num_queries, num_predictions)
        )
        return output_masks, iou_predictions
    
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def get_rescaled_pts(self, batched_points: torch.Tensor, input_h: int, input_w: int):
        return torch.stack(
            [
                torch.where(
                    batched_points[..., 0] >= 0,
                    batched_points[..., 0] * self.image_encoder.img_size / input_w,
                    -1.0,
                ),
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * self.image_encoder.img_size / input_h,
                    -1.0,
                ),
            ],
            dim=-1,
        )

    @torch.jit.export
    def get_image_embeddings(self, batched_images) -> torch.Tensor:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_images: A tensor of shape [B, 3, H, W]
        Returns:
          List of image embeddings each of of shape [B, C(i), H(i), W(i)].
          The last embedding corresponds to the final layer.
        """
        batched_images = self.preprocess(batched_images)
        return self.image_encoder(batched_images)
    
    # def encoder_image_embeddings(self, images: List[torch.Tensor],):
    #     input_images = torch.stack([self.preprocess(x) for x in images], dim=0)
    #     image_embeddings = self.image_encoder(input_images)
    #     return image_embeddings

    def encoder_image_embeddings(self, images: List[torch.Tensor],):
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_images: A tensor of shape [B, 3, H, W]
        Returns:
          List of image embeddings each of of shape [B, C(i), H(i), W(i)].
          The last embedding corresponds to the final layer.
        """
        batched_images = torch.stack(images, dim=0)
        batched_images = self.preprocess(batched_images)
        return self.image_encoder(batched_images)

    def forward(
        self,
        batched_images: torch.Tensor,
        batched_points: torch.Tensor,
        batched_point_labels: torch.Tensor,
        scale_to_original_image_size: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_images: A tensor of shape [B, 3, H, W]
          batched_points: A tensor of shape [B, num_queries, max_num_pts, 2]
          batched_point_labels: A tensor of shape [B, num_queries, max_num_pts]

        Returns:
          A list tuples of two tensors where the ith element is by considering the first i+1 points.
            low_res_mask: A tensor of shape [B, 256, 256] of predicted masks
            iou_predictions: A tensor of shape [B, max_num_queries] of estimated IOU scores
        """
        batch_size, _, input_h, input_w = batched_images.shape
        image_embeddings = self.get_image_embeddings(batched_images)
        return self.predict_masks(
            image_embeddings,
            batched_points,
            batched_point_labels,
            multimask_output=True,
            input_h=input_h,
            input_w=input_w,
            output_h=input_h if scale_to_original_image_size else -1,
            output_w=input_w if scale_to_original_image_size else -1,
        )
    
    # def forward_train(
    #     self,
    #     batched_input: List[Dict[str, Any]],
    #     multimask_output: bool,
    #     image_size: Tuple[int, ...],
    #     input_image_embeddings: torch.Tensor = None,
    # ) -> List[Dict[str, torch.Tensor]]:

    #     image_embeddings = input_image_embeddings
    #     outputs = []
    #     for image_record, curr_embedding in zip(batched_input, image_embeddings):
    #         points = (image_record["point_coords"], image_record["point_labels"])
    #         sparse_embeddings, dense_embeddings = self.prompt_encoder(
    #             points=points,
    #             boxes=image_record.get("boxes", None),
    #             masks=image_record.get("mask_inputs", None),
    #         )
    #         # coords, labels = points
    #         # sparse_embeddings = self.prompt_encoder(coords, labels)
    #         # dense_embeddings = self.prompt_encoder.get_dense_pe()

    #         low_res_masks, iou_predictions = self.mask_decoder(
    #             image_embeddings=curr_embedding.unsqueeze(0),
    #             image_pe=self.prompt_encoder.get_dense_pe(),
    #             sparse_prompt_embeddings=sparse_embeddings,
    #             dense_prompt_embeddings=dense_embeddings,
    #             multimask_output=multimask_output,
    #         )
    #         masks = self.postprocess_masks(
    #             low_res_masks,
    #             input_size=image_size,
    #             original_size=image_record["original_size"],
    #         )
    #         masks = masks > self.mask_threshold
    #         outputs.append(
    #             {
    #                 "masks": masks,
    #                 "iou_predictions": iou_predictions,
    #                 "low_res_logits": low_res_masks,
    #             }
    #         )
    #     return outputs

    def forward_train(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
        image_size: Tuple[int, ...],
        input_image_embeddings: torch.Tensor = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Forward pass during training for batched inputs.

        Args:
            batched_input: A list of dicts, each containing:
                - "point_coords": [num_queries, num_points, 2]
                - "point_labels": [num_queries, num_points]
                - "original_size": Tuple[int, int]
            multimask_output: Whether to output multiple masks per prompt.
            image_size: The input image size (H, W) before resizing.
            input_image_embeddings: Optional precomputed image embeddings 
                of shape [B, C, H, W]. If not provided, compute from image input.

        Returns:
            List of dicts for each image with keys:
                - "masks": Thresholded masks (bool)
                - "iou_predictions": IoU scores
                - "low_res_logits": Raw low-res logits
        """
        outputs = []

        if input_image_embeddings is None:
            images = torch.stack([record["image"] for record in batched_input], dim=0)
            input_image_embeddings = self.get_image_embeddings(images)

        batch_size = len(batched_input)
        device = input_image_embeddings.device

        # Gather prompts
        # batched_points = torch.stack([rec["point_coords"] for rec in batched_input]).to(device)
        batched_points = torch.cat([rec["point_coords"] for rec in batched_input], dim=0).to(device)
        # batched_point_labels = torch.stack([rec["point_labels"] for rec in batched_input]).to(device)
        batched_point_labels = torch.cat([rec["point_labels"] for rec in batched_input], dim=0).to(device)

        input_h, input_w = image_size

        masks, iou_predictions = self.predict_masks(
            input_image_embeddings,
            batched_points,
            batched_point_labels,
            multimask_output=multimask_output,
            input_h=input_h,
            input_w=input_w,
            output_h=input_h,
            output_w=input_w,
        )

        for i, rec in enumerate(batched_input):
            masks_i = masks[i] > self.mask_threshold
            outputs.append({
                "masks": masks_i,
                "iou_predictions": iou_predictions[i],
                "low_res_logits": masks[i],  # raw logits
            })

        return outputs

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # if (
        #     x.shape[2] != self.image_encoder.img_size
        #     or x.shape[3] != self.image_encoder.img_size
        # ):
        if (
            x.shape[2] != self.image_encoder.img_size
            or x.shape[1] != self.image_encoder.img_size
        ):
            x = F.interpolate(
                x,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
            )
        return (x - self.pixel_mean) / self.pixel_std


def build_efficient_sam(encoder_patch_embed_dim, encoder_num_heads, image_size=1024, checkpoint=None):
    img_size = image_size
    encoder_patch_size = 16
    encoder_depth = 12
    encoder_mlp_ratio = 4.0
    encoder_neck_dims = [256, 256]
    decoder_max_num_input_points = 6
    decoder_transformer_depth = 2
    decoder_transformer_mlp_dim = 2048
    decoder_num_heads = 8
    decoder_upscaling_layer_dims = [64, 32]
    num_multimask_outputs = 3
    iou_head_depth = 3
    iou_head_hidden_dim = 256
    activation = "gelu"
    normalization_type = "layer_norm"
    normalize_before_activation = False

    assert activation == "relu" or activation == "gelu"
    if activation == "relu":
        activation_fn = nn.ReLU
    else:
        activation_fn = nn.GELU

    image_encoder = ImageEncoderViT(
        img_size=img_size,
        patch_size=encoder_patch_size,
        in_chans=3,
        patch_embed_dim=encoder_patch_embed_dim,
        normalization_type=normalization_type,
        depth=encoder_depth,
        num_heads=encoder_num_heads,
        mlp_ratio=encoder_mlp_ratio,
        neck_dims=encoder_neck_dims,
        act_layer=activation_fn,
    )

    image_embedding_size = image_encoder.image_embedding_size
    encoder_transformer_output_dim = image_encoder.transformer_output_dim

    sam = EfficientSam(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=encoder_transformer_output_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(img_size, img_size),
        ),
        decoder_max_num_input_points=decoder_max_num_input_points,
        mask_decoder=MaskDecoder(
            transformer_dim=encoder_transformer_output_dim,
            transformer=TwoWayTransformer(
                depth=decoder_transformer_depth,
                embedding_dim=encoder_transformer_output_dim,
                num_heads=decoder_num_heads,
                mlp_dim=decoder_transformer_mlp_dim,
                activation=activation_fn,
                normalize_before_activation=normalize_before_activation,
            ),
            num_multimask_outputs=num_multimask_outputs,
            activation=activation_fn,
            normalization_type=normalization_type,
            normalize_before_activation=normalize_before_activation,
            iou_head_depth=iou_head_depth - 1,
            iou_head_hidden_dim=iou_head_hidden_dim,
            upscaling_layer_dims=decoder_upscaling_layer_dims,
        ),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
    )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        sam.load_state_dict(state_dict["model"])
    return sam
