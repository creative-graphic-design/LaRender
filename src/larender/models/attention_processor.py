import math
from enum import Enum, auto
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from diffusers.models.attention_processor import Attention, AttnProcessor


class DensitySchedulerType(Enum):
    unchanged = auto()
    opaque = auto()
    inverse_proportional = auto()


class LaRenderAttnProcessor(AttnProcessor):
    def _get_multipied_densities(
        self,
        density_scheduler_type: DensitySchedulerType,
        densities,
        current_count: int,
        num_inference_steps: int,
    ) -> List[np.float64]:
        if density_scheduler_type is DensitySchedulerType.inverse_proportional:
            density_multiplier = num_inference_steps / (current_count + 1)
            return [d * density_multiplier for d in densities]
        elif density_scheduler_type is DensitySchedulerType.opaque:
            return [d * num_inference_steps for d in densities]
        else:
            return densities

    def _bboxes_to_mask(self, bboxes, h, w, dtype: torch.dtype, device: torch.device):
        """
        Args:
            bboxes (List[List[y1, y2, x1, x2]]):
                The bounding boxes in y1, y2, x1, x2 format.
        """
        masks = []
        for bbox in bboxes:
            M = torch.zeros((h, w), dtype=dtype, device=device)
            upper = int(h * bbox[0])
            lower = int(h * bbox[1])
            left = int(w * bbox[2])
            right = int(w * bbox[3])
            assert lower > upper and right > left, f"Invalid bounding box: {bbox}"
            M[upper:lower, left:right] = 1
            masks.append(M)
        return masks

    def _cross_attention(self, attn: Attention, hidden_states, encoder_hidden_states):
        assert (
            attn.to_q is not None
            and attn.to_k is not None
            and attn.to_v is not None
            and attn.to_out is not None
        )

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states, attention_probs

    def __call__(  # type: ignore[override]
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        transmittance_use_bbox: bool,
        transmittance_use_attn_map: bool,
        objects: Sequence[str],
        indices: Sequence[List[int]],
        locations: Sequence[Tuple[float, float, float, float]],
        densities,
        density_scheduler_type: DensitySchedulerType,
        width: int,
        height: int,
        current_count: int,
        num_inference_steps: int,
        token_max_length: int,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            return super().__call__(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                temb,
                *args,
                **kwargs,
            )

        num_objects = len(objects)

        latent_h = int(math.sqrt(hidden_states.shape[1] * height / width))
        assert hidden_states.size()[1] % latent_h == 0, (
            "Cannot infer latent_h and latent_w, please check the code."
        )
        latent_w = int(hidden_states.size()[1] / latent_h)

        densities_multiplied = self._get_multipied_densities(
            density_scheduler_type,
            densities,
            current_count,
            num_inference_steps=num_inference_steps,
        )

        assert encoder_hidden_states.shape[1] % token_max_length == 0

        # object-wise cross-attention and transmittance map
        R = []
        dtype = hidden_states.dtype
        device = hidden_states.device
        if transmittance_use_bbox:
            M = self._bboxes_to_mask(locations, latent_h, latent_w, dtype, device)
        else:
            M = [
                torch.ones((latent_h, latent_w), dtype=dtype, device=device)
                for _ in locations
            ]

        for i in range(num_objects):
            context = encoder_hidden_states[
                :, i * token_max_length : (i + 1) * token_max_length, :
            ]
            R_i, attn_probs = self._cross_attention(attn, hidden_states, context)
            R.append(R_i.reshape(R_i.shape[0], latent_h, latent_w, R_i.shape[2]))

            if transmittance_use_attn_map:
                attn_map = attn_probs.reshape(
                    attn_probs.shape[0], latent_h, latent_w, -1
                )
                attn_map = attn_map[..., indices[i]].mean(dim=-1).mean(dim=0)
                min_val = attn_map.min()
                max_val = attn_map.max()
                normalized_attn_map = (attn_map - min_val) / (max_val - min_val)
                M[i] *= normalized_attn_map

        # accumulated transmittance maps (visibility of planar i from the virtual camera)
        T = [
            torch.ones((latent_h, latent_w), dtype=dtype, device=device)
            for _ in range(num_objects)
        ]
        for i in range(num_objects - 1, -1, -1):  # top to bottom
            for j in range(i + 1, num_objects):
                # TODO
                T[i] *= torch.exp(-torch.tensor(densities_multiplied[j]) * M[j])

        # rendering
        S: Optional[torch.Tensor] = None
        R_out: Optional[torch.Tensor] = None

        for i in range(num_objects):
            contrib = T[i] * (1 - math.exp(-densities_multiplied[i])) * M[i]
            assert isinstance(contrib, torch.Tensor)
            contrib = contrib[None, :, :, None]  # unsqueeze 0, -1
            if R_out is None:
                R_out = contrib * R[i]
                S = contrib
            else:
                assert isinstance(R_out, torch.Tensor) and isinstance(S, torch.Tensor)
                R_out += contrib * R[i]
                S += contrib

        assert isinstance(R_out, torch.Tensor) and isinstance(S, torch.Tensor)
        S = torch.clamp(S, min=1e-6)
        R_out /= S

        return R_out.reshape(*hidden_states.shape)
