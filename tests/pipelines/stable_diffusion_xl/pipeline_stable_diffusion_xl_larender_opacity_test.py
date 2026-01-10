from typing import List, Tuple

import pytest
import torch
from pytest_lazy_fixtures import lf


@pytest.fixture
def sample_1() -> Tuple[
    List[str], List[Tuple[float, float, float, float]], List[float]
]:
    objects: List[str] = [
        "the entrance of a convenience store",
        "a clear glass door",
    ]
    locations: List[Tuple[float, float, float, float]] = [
        (0.1, 0.9, 0.1, 0.9),
        (0.3, 0.8, 0.25, 0.75),
    ]
    opacity: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    return objects, locations, opacity


@pytest.mark.parametrize(
    argnames=("objects_locations_opacities"),
    argvalues=(
        lf("sample_1"),
        lf("sample_2"),
        lf("sample_3"),
        lf("sample_4"),
        lf("sample_5"),
        lf("sample_6"),
        lf("sample_7"),
        lf("sample_8"),
        lf("sample_9"),
        lf("sample_10"),
        lf("sample_11"),
        lf("sample_12"),
        lf("sample_13"),
        lf("sample_14"),
        lf("sample_15"),
        lf("sample_16"),
        lf("sample_17"),
        lf("sample_18"),
        lf("sample_19"),
        lf("sample_20"),
        lf("sample_21"),
        lf("sample_22"),
    ),
)
@pytest.mark.parametrize(
    argnames="model_id",
    argvalues=(
        "comin/IterComp",
        "jiuntian/gligen-xl-1024",
    ),
)
def test_pipelie_stable_diffusion_xl_larender_opacity(
    model_id: str,
    device: torch.device,
    torch_dtype: torch.dtype,
    objects_locations_opacities: Tuple[
        List[str], List[Tuple[float, float, float, float]], List[float]
    ],
):
    objects, locations, opacity = objects_locations_opacities

    pipe = StableDiffusionXLLaRenderPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )

    raise NotImplementedError
