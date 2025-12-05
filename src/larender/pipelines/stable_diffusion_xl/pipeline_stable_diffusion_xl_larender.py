from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import spacy
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.models import UNet2DConditionModel
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils.logging import get_logger

from larender.models.attention_processor import (
    DensitySchedulerType,
    LaRenderAttnProcessor,
)

logger = get_logger(__name__)


@dataclass
class DependencyParser(object):
    nlp: spacy.language.Language = spacy.load("en_core_web_sm")

    def __call__(self, text: str, never_empty: bool) -> Tuple[List[int], List[str]]:
        doc = self.nlp(text)
        indices, subjects = [], []

        for token in doc:
            # noun of a sentence or noun in a phrase
            cond1 = token.dep_ in ["nsubj", "nsubjpass"]
            cond2 = token.dep_ == "ROOT" and token.pos_ in ["NOUN", "PROPN"]
            if cond1 or cond2:
                indices.append(token.i)
                subjects.append(token.text)

        if never_empty and len(indices) == 0:
            indices.append(0)
            subjects.append(doc[0].text)

        return indices, subjects


class StableDiffusionXLLaRenderPipeline(StableDiffusionXLPipeline):
    unet: UNet2DConditionModel
    scheduler: DPMSolverMultistepScheduler

    def encode_objects_prompt(
        self,
        objects: Sequence[str],
        negative_prompt,
        # prompt: str,
        # prompt_2: Optional[str] = None,
        # device: Optional[torch.device] = None,
        # num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        # negative_prompt: Optional[str] = None,
        # negative_prompt_2: Optional[str] = None,
        # prompt_embeds: Optional[torch.Tensor] = None,
        # negative_prompt_embeds: Optional[torch.Tensor] = None,
        # pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # lora_scale: Optional[float] = None,
        # clip_skip: Optional[int] = None,
    ):
        # return super().encode_prompt(
        #     prompt,
        #     prompt_2,
        #     device,
        #     num_images_per_prompt,
        #     do_classifier_free_guidance,
        #     negative_prompt,
        #     negative_prompt_2,
        #     prompt_embeds,
        #     negative_prompt_embeds,
        #     pooled_prompt_embeds,
        #     negative_pooled_prompt_embeds,
        #     lora_scale,
        #     clip_skip,
        # )
        prompt_embeds_list: List[torch.Tensor] = []
        negative_prompt_embeds_list: List[torch.Tensor] = []
        pooled_prompt_embeds_list: List[torch.Tensor] = []
        negative_pooled_prompt_embeds_list: List[torch.Tensor] = []

        for obj in objects:
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.encode_prompt(
                prompt=obj,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )
            prompt_embeds_list.append(prompt_embeds)
            negative_prompt_embeds_list.append(negative_prompt_embeds)
            pooled_prompt_embeds_list.append(pooled_prompt_embeds)
            negative_pooled_prompt_embeds_list.append(negative_pooled_prompt_embeds)

        prompt_embeds = torch.cat(prompt_embeds_list, dim=1)
        negative_prompt_embeds = torch.cat(negative_prompt_embeds_list, dim=1)
        pooled_prompt_embeds = sum(pooled_prompt_embeds_list) / len(
            pooled_prompt_embeds
        )
        negative_pooled_prompt_embeds = sum(negative_pooled_prompt_embeds_list) / len(
            negative_pooled_prompt_embeds
        )
        (
            _,
            _,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=",".join(objects),
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    @torch.no_grad()
    def __call__(
        self,
        objects: Sequence[str],
        locations: Sequence[Tuple[float, float, float, float]],
        alpha: Union[float, Sequence[float]] = 0.8,
        use_dependency_parsing: bool = True,
        ignore_dp_failure: bool = True,
        density_scheduler_type: DensitySchedulerType = DensitySchedulerType.unchanged,
        transmittance_use_bbox: bool = True,
        transmittance_use_attn_map: bool = True,
        #
        # prompt: Union[str, List[str]] = None,
        # prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        self.unet.set_attn_processor(LaRenderAttnProcessor())

        if not isinstance(alpha, Sequence):
            alpha = [alpha for _ in range(len(objects))]
        assert all(0 <= a < 1 for a in alpha), "alpha values must be in [0, 1)"
        densities = [-np.log(1 - a) for a in alpha]

        if use_dependency_parsing:
            parser = DependencyParser()
            indices, subjects = zip(
                *[parser(obj, ignore_dp_failure) for obj in objects]
            )
            assert all(len(ind) > 0 for ind in indices), (
                f"Dependency Parsing error: subject indices contain empty cases, please rephrase your prompts or set ignore_dp_failure=True, indices: {indices}"
            )
        else:
            raise NotImplementedError

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_objects_prompt(
            objects=objects,
            negative_prompt=negative_prompt,
        )

        cross_attention_kwargs = cross_attention_kwargs or {}

        step_index = self.scheduler.step_index  # type: ignore[attr-defined]
        current_count = step_index if step_index is not None else 0
        cross_attention_kwargs.update(
            {
                "transmittance_use_bbox": transmittance_use_bbox,
                "transmittance_use_attn_map": transmittance_use_attn_map,
                "density_scheduler_type": density_scheduler_type,
                "objects": objects,
                "locations": locations,
                "indices": indices,
                "densities": densities,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "current_count": current_count,
                "token_max_length": self.tokenizer.model_max_length,
            }
        )
        return super().__call__(
            # prompt=prompt,
            # prompt_2=prompt_2,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            sigmas=sigmas,
            denoising_end=denoising_end,
            guidance_scale=guidance_scale,
            # negative_prompt=negative_prompt,
            # negative_prompt_2=negative_prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            ip_adapter_image=ip_adapter_image,
            ip_adapter_image_embeds=ip_adapter_image_embeds,
            output_type=output_type,
            return_dict=return_dict,
            cross_attention_kwargs=cross_attention_kwargs,
            guidance_rescale=guidance_rescale,
            original_size=original_size,
            crops_coords_top_left=crops_coords_top_left,
            target_size=target_size,
            negative_original_size=negative_original_size,
            negative_crops_coords_top_left=negative_crops_coords_top_left,
            negative_target_size=negative_target_size,
            clip_skip=clip_skip,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            **kwargs,
        )
