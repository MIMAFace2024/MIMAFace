#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import cv2
import os
import itertools
import random
import shutil
from pathlib import Path
import numpy as np
import accelerate
from einops import rearrange
from kornia.geometry.transform import warp_affine
import types

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.utils as ttf
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPVisionModel, CLIPImageProcessor

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
    StableDiffusionPipeline
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from dataset.datasets_video import DatasetDriveVideo  
from models.mimaface import MIMAFacePostfuseModule, CLIPImageEncoder
from peft import LoraConfig, get_peft_model, PeftModel

from models.D3DFR.D3DFR_ReconModel import D3DFR_wrapper
from models.ArcFace.CurricularFace_wrapper import CurricularFaceOurCrop
from pipeline_image import MIMAfacePipeline

from dataset import face_util 
from util import parse_args, save_videos_img2mp4, convert_batch_to_nprgb



# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0.dev0")

logger = get_logger(__name__)

pil2tensor = transforms.Compose([ 
    transforms.ToTensor(), 
    transforms.Normalize(mean=0.5, std=0.5)])
clip_image_processor = CLIPImageProcessor()



def convert_batch_to_nprgb(batch, nrow):
    grid_tensor = ttf.make_grid(batch * 0.5 + 0.5, nrow=nrow)
    im_rgb = (255 * grid_tensor.permute(1, 2, 0).cpu().numpy()).astype('uint8')
    return im_rgb    


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid



def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="training script.")
    parser.add_argument(
        "--dataset_base_path",
        type=str,
        default="/apdcephfs/share_1307729/shared_info/multi_modal_human_texture_generation/dataset/",
        help="dataset list, prompt-image pair",
    )
    parser.add_argument(
        "--dataset_list_path",
        type=str,
        default=None,
        help="dataset list, prompt-image pair",
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_clip_name_or_path",
        type=str,
        default="laion/CLIP-ViT-H-14-laion2B-s32B-b79",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_image2token",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--pretrained_id2token",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--pretrained_net_seg_res18",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--pretrained_unet_path",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--pretrained_motion_module_path",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/youtu_xuanyuan_shuzhiren_2906355_cq10/private/junweizhu/huggingface/hub",
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--local_files_only",
        default=False,
        action="store_true",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--video_length", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    
    # lora args
    parser.add_argument("--use_lora", action="store_true", help="Whether to use Lora for parameter efficient tuning")
    parser.add_argument("--lora_r", type=int, default=128, help="Lora rank, only used if use_lora is True")
    parser.add_argument("--lora_alpha", type=int, default=128, help="Lora alpha, only used if use_lora is True")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Lora dropout, only used if use_lora is True")
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        help="Bias type for Lora. Can be 'none', 'all' or 'lora_only', only used if use_lora is True",
    )
    
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--pretrained_MIM",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./',
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1000,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )


    return args


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=args.cache_dir, local_files_only=args.local_files_only)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant, cache_dir=args.cache_dir, local_files_only=args.local_files_only
    )
    if args.pretrained_unet_path is not None:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
        )
        in_channels = 13
        out_channels = unet.conv_in.out_channels
        unet.register_to_config(in_channels=in_channels)

        with torch.no_grad():
            new_conv_in = torch.nn.Conv2d(
                in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
            unet.conv_in = new_conv_in
        
        unet.load_state_dict(torch.load(args.pretrained_unet_path)) # @@@
    else:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
        )
    
        in_channels = 13
        out_channels = unet.conv_in.out_channels
        unet.register_to_config(in_channels=in_channels)

        with torch.no_grad():
            new_conv_in = torch.nn.Conv2d(
                in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
            unet.conv_in = new_conv_in


    net_vision_encoder = CLIPVisionModel.from_pretrained(args.pretrained_clip_name_or_path)
    net_vision_encoder.vision_model.post_layernorm.requires_grad_(False)
    
    # net_vision_encoder = CLIPImageEncoder.from_pretrained(args.pretrained_clip_name_or_path)
    # net_vision_encoder.load_state_dict(torch.load('/youtu_xuanyuan_shuzhiren_2906355_cq10/private/hyy/mimaface-train/clip/clip.pth')) # @@@
    # image_encoder.vision_model.post_layernorm.requires_grad_(False)

    mim = MIMAFacePostfuseModule(768)
    if args.pretrained_MIM is not None:
        info = mim.load_state_dict(torch.load(args.pretrained_MIM), strict=False)
        print(args.pretrained_MIM, info)
        

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model


    D3DFR_inst = D3DFR_wrapper()
    curricularface_inst = CurricularFaceOurCrop()

    vae.requires_grad_(False)
    curricularface_inst.requires_grad_(False)
    curricularface_inst = curricularface_inst.eval()
    D3DFR_inst.requires_grad_(False)
    D3DFR_inst = D3DFR_inst.eval()
    
    # net_vision_encoder.requires_grad_(True)
    net_vision_encoder.vision_model.requires_grad_(True)
    net_vision_encoder.vision_model.post_layernorm.requires_grad_(False)
    mim.requires_grad_(True)
    

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()


    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )



    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    # Set unet trainable parameters
    unet.requires_grad_(True)
    # trainable_modules = ["motion_modules.", "conv_in."]
    # for name, param in unet.named_parameters():
    #     for trainable_module_name in trainable_modules:
    #         if trainable_module_name in name:
    #             param.requires_grad = True
         
    # params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    params_to_optimize = itertools.chain(unet.parameters(), mim.parameters(), net_vision_encoder.parameters())

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    clip_image_processor = CLIPImageProcessor()
    train_dataset = DatasetDriveVideo(args.dataset_base_path, args.dataset_list_path, args.resolution, clip_image_processor, train=True, video_length=args.video_length)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    # net_seg_res18 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_seg_res18)
    net_vision_encoder.train()
    mim.train()
    unet.train()
    unet, mim, net_vision_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    unet, mim, net_vision_encoder, optimizer, train_dataloader, lr_scheduler
)
    for name, param in unet.named_parameters():
        if param.requires_grad:
            logger.info(f"Trainable parameter: {name} with shape {param.shape}")
    for name, param in mim.named_parameters():
        if param.requires_grad:
            logger.info(f"Trainable parameter: {name} with shape {param.shape}")
    for name, param in net_vision_encoder.named_parameters():
        if param.requires_grad:
            logger.info(f"Trainable parameter: {name} with shape {param.shape}")
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    # net_vision_encoder.to(accelerator.device, dtype=weight_dtype)
    # mim.to(accelerator.device, dtype=weight_dtype)
    
    D3DFR_inst.to(accelerator.device)
    curricularface_inst.to(accelerator.device)

    # empty_prompt_token = torch.load('empty_prompt_embedding.pth').view(1, 77,768).to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        # accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0


    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                # process input
                # import pdb;pdb.set_trace()
                src_img = batch['image_src'] # torch.Size([2, 3, 512, 512])
                src_img_crop256 = batch['image_src_crop256'] # torch.Size([2, 3, 256, 256])
                clip_input_tensors = batch['image_src_clip']
                drv_same_img = rearrange(batch['image_tar'], "b f c h w -> (b f) c h w")
                drv_same_img_crop256 = rearrange(batch['image_tar_crop256'], "b f c h w -> (b f) c h w")
                
                src_wapmat = batch['image_src_warpmat256'] # torch.Size([2, 2, 3])
                src_inv_wapmat = batch['image_src_inverse_warpmat256'] # torch.Size([2, 2, 3])
                drv_same_wapmat = rearrange(batch['image_tar_warpmat256'], "b f c r -> (b f) c r") # torch.Size([2, 12, 2, 3])
                drv_same_inv_wapmat = rearrange(batch['image_tar_inverse_warpmat256'], "b f c r -> (b f) c r")

                ref_img = drv_same_img
                ref_img_crop256 = drv_same_img_crop256 
                ref_wapmat = drv_same_wapmat
                ref_inv_wapmat = drv_same_inv_wapmat

                D3DFR_inst = D3DFR_inst.eval()
                
                temp = D3DFR_inst.forward_uncropped_tensor_FaceDrive_WithWarpmat_Reenact(#_Reenact512
                    rearrange(src_img.unsqueeze(1).repeat(1,args.video_length,1,1,1),  "b f c h w -> (b f) c h w"), 
                    rearrange(src_wapmat.unsqueeze(1).repeat(1,args.video_length,1,1), "b f c r -> (b f) c r").to(torch.float32),
                    rearrange(src_inv_wapmat.unsqueeze(1).repeat(1,args.video_length,1,1), "b f c r -> (b f) c r").to(torch.float32), 
                    ref_img, 
                    ref_wapmat.to(torch.float32), 
                    ref_inv_wapmat.to(torch.float32)
                )
                id25088 = curricularface_inst.deep_features(warp_affine(src_img, src_wapmat.to(torch.float32), dsize=(args.resolution, args.resolution))).flatten(1).to(dtype=weight_dtype)

                imRef_d3dBlendSrc = F.interpolate(temp['imRef_d3dBlendSrc'], size=(args.resolution, args.resolution), mode="bilinear", align_corners=False,).to(dtype=weight_dtype) 
                coeffs = temp['coeffBlendSrcRef'].to(dtype=weight_dtype)
                id_coeff = coeffs[:, :80]  # identity(shape) coeff of dim 80
                exp_coeff = coeffs[:, 80:144]  # expression coeff of dim 64
                tex_coeff = coeffs[:, 144:224]  # texture(albedo) coeff of dim 80
                # ruler angles(x,y,z) for rotation of dim 3
                angles = coeffs[:, 224:227]
                # lighting coeff for 3 channel SH function of dim 27
                gamma = coeffs[:, 227:254]
                translation = coeffs[:, 254:257]  # translation coeff of dim 3
                trg_gaze = coeffs[:, 257:]
            
                latents = vae.encode(ref_img.to(dtype=weight_dtype)).latent_dist.mode()
                latents = latents * vae.config.scaling_factor
                render_latents = vae.encode(imRef_d3dBlendSrc.to(dtype=weight_dtype)).latent_dist.mode()
                render_latents = render_latents * vae.config.scaling_factor
                
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            object_embeds257 = net_vision_encoder(clip_input_tensors.to(dtype=weight_dtype)).last_hidden_state                
            
            object_hidden_states = mim(
                object_embeds=object_embeds257.repeat(1,args.video_length,1,1), 
                arcface_embeds=rearrange(id25088.unsqueeze(1).repeat(1,args.video_length,1),"b f d -> (b f) d"),   
                gaze_embeds=trg_gaze, 
                exp_embeds=exp_coeff, 
                tex_embeds=tex_coeff, 
                gamma_embeds=gamma, 
                angles_embeds=angles,
                translation_embeds=translation
            ) # torch.Size([24, 258, 768])
            
            rand_num1 = random.random()
            if rand_num1 < 0.01:
                gt_image_latents = latents
                gt_mask = torch.ones_like(latents)[:,:1]
            else:
                gt_image_latents = torch.zeros_like(latents)
                gt_mask = torch.zeros_like(latents)[:,:1]
            

            rand_num2 = random.random()
            if rand_num2 < 0.1:
                object_hidden_states = torch.zeros_like(object_hidden_states)

            encoder_hidden_states = object_hidden_states

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )
            # torch.save(render_latents,'render_latents.pth')
            # torch.save(encoder_hidden_states,'encoder_hidden_states.pth')
            # import pdb;pdb.set_trace()
            pred = unet(
                torch.cat([noisy_latents, render_latents, gt_image_latents, gt_mask], dim=1), # torch.Size([2, 13, 12, 64, 64])
                timesteps, # 2
                encoder_hidden_states # torch.Size([2, 258, 768])
                ).sample # torch.Size([2, 4, 12, 64, 64])
            # import pdb;pdb.set_trace()

            debug = False
            if debug and step%5==1:

                from diff import GaussianDiffusion
                gd = GaussianDiffusion(1000)
                est_x_0 = gd.predict_start_from_noise(noisy_latents[:1], timesteps[:1], pred[:1])
                # import pdb;pdb.set_trace()
                pred_x0_sample = vae.decode(est_x_0.to(weight_dtype) / vae.config.scaling_factor, return_dict=False)[0]

                vis_image = np.asarray(pred_x0_sample.squeeze().permute(1, 2, 0).detach().cpu())
                max_value = vis_image.max()
                min_value = vis_image.min()
                vis_image = (vis_image - min_value) * 256 / (max_value - min_value) 
                vis_image = np.uint8(vis_image)

                vis_images = np.asarray(src_img[0].permute(1, 2, 0).detach().cpu())
                max_values = vis_images.max()
                min_values = vis_images.min()
                vis_images = (vis_images - min_values) * 256 / (max_values - min_values) 
                vis_images = np.uint8(vis_images)

                # vis_image = np.uint8((vis_image+1)*127.5)
                vis_image = Image.fromarray(np.concatenate([vis_image,vis_images],axis=1))
                vis_image.save(f"train{int(timesteps[:1].detach().cpu())}.jpg")


            
            if args.snr_gamma is None:
                loss = F.mse_loss(pred.float(), target.float(), reduction="mean")    ### + 2*F.mse_loss(pred_mask, gen_mask_gt)
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.

                snr_timesteps = timesteps
                snr = compute_snr(noise_scheduler, snr_timesteps)
                base_weight = (
                    torch.stack([snr, args.snr_gamma * torch.ones_like(snr_timesteps)], dim=1).min(dim=1)[0] / snr
                )

                if noise_scheduler.config.prediction_type == "v_prediction":
                    # Velocity objective needs to be floored to an SNR weight of one.
                    mse_loss_weights = base_weight + 1
                else:
                    # Epsilon and sample both use the same loss weights.
                    mse_loss_weights = base_weight

                loss = F.mse_loss(pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean() #####+ 2*F.mse_loss(pred_mask, gen_mask_gt)

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = params_to_optimize
                # params_to_clip = params_to_optimize
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 1000:
                        unwrap_unet = unwrap_model(unet)
                        unwrap_unet.save_pretrained(os.path.join(args.output_dir, 'pretrained_unet'))
                        
                        unwrap_mim = unwrap_model(mim)
                        torch.save(unwrap_mim.state_dict(), os.path.join(args.output_dir, "mim.pth"), _use_new_zipfile_serialization=False)
                        
                        unwrap_net_vision_encoder = unwrap_model(net_vision_encoder)
                        unwrap_net_vision_encoder.save_pretrained(os.path.join(args.output_dir, 'vision_encoder'))
                        
                    
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()


    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
    