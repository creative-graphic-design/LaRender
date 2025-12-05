from typing import List, Tuple

import pytest
import torch
from diffusers.schedulers import DPMSolverMultistepScheduler
from pytest_lazy_fixtures import lf

from larender.pipelines import StableDiffusionXLLaRenderPipeline


@pytest.fixture
def sample_1() -> Tuple[List[str], List[Tuple[float, float, float, float]]]:
    objects: List[str] = [
        "a large empty living room",
        "a brown piano",
        "a giraffe is standing next to a piano and a cat",
        "a sofa",
        "a ginger cat sitting",
        "a blue teddy bear next to a cat",
    ]
    locations: List[Tuple[float, float, float, float]] = [
        (0, 1, 0, 1),
        (0.4, 0.7, 0.2, 0.9),
        (0, 0.8, 0, 0.5),
        (0.6, 0.9, 0.05, 0.95),
        (0.5, 0.75, 0.5, 0.7),
        (0.5, 0.9, 0.7, 0.95),
    ]  # mode: [y1, y2, x1, x2], range: 0~1
    return objects, locations


@pytest.fixture
def sample_2() -> Tuple[List[str], List[Tuple[float, float, float, float]]]:
    objects: List[str] = [
        "forest",
        "a white cat standing on the ground next to a branch",
        "a brown dog sitting on the ground next to a branch",
        "a long branch",
    ]
    locations: List[Tuple[float, float, float, float]] = [
        (0, 1, 0, 1),
        (0.3, 0.8, 0.1, 0.4),
        (0.3, 0.8, 0.5, 0.8),
        (0.5, 0.6, 0, 1),
    ]
    return objects, locations


@pytest.fixture
def sample_3() -> Tuple[List[str], List[Tuple[float, float, float, float]]]:
    objects: List[str] = [
        "forest",
        "a white cat standing on the ground next to a branch",
        "a long branch",
        "a brown dog sitting next to a branch",
    ]
    locations: List[Tuple[float, float, float, float]] = [
        [0, 1, 0, 1],
        [0.3, 0.8, 0.1, 0.4],
        [0.5, 0.6, 0, 1],
        [0.3, 0.8, 0.5, 0.8],
    ]
    return objects, locations


@pytest.fixture
def sample_4() -> Tuple[List[str], List[Tuple[float, float, float, float]]]:
    objects: List[str] = [
        "forest",
        "a brown dog sitting",
        "a long branch",
        "a white cat standing",
    ]
    locations: List[Tuple[float, float, float, float]] = [
        [0, 1, 0, 1],
        [0.3, 0.8, 0.5, 0.8],
        [0.5, 0.6, 0, 1],
        [0.3, 0.8, 0.1, 0.4],
    ]
    return objects, locations


@pytest.fixture
def sample_5() -> Tuple[List[str], List[Tuple[float, float, float, float]]]:
    objects: List[str] = [
        "forest",
        "a long branch",
        "a brown dog sitting",
        "a white cat standing",
    ]
    locations: List[Tuple[float, float, float, float]] = [
        [0, 1, 0, 1],
        [0.5, 0.6, 0, 1],
        [0.3, 0.8, 0.55, 0.8],
        [0.3, 0.8, 0.2, 0.4],
    ]
    return objects, locations


@pytest.fixture
def sample_6() -> Tuple[List[str], List[Tuple[float, float, float, float]]]:
    objects: List[str] = [
        "a giraffe",
        "an airplane on the ground",
    ]
    locations: List[Tuple[float, float, float, float]] = [
        [0.2, 0.8, 0.4, 0.6],
        [0.5, 0.8, 0, 1],
    ]
    return objects, locations


@pytest.fixture
def sample_7() -> Tuple[List[str], List[Tuple[float, float, float, float]]]:
    objects: List[str] = [
        "a house",
        "lawn",
        "a giant moon",
    ]
    locations: List[Tuple[float, float, float, float]] = [
        [0.3, 0.7, 0.1, 0.9],
        [0.7, 1, 0, 1],
        [0.55, 0.8, 0.2, 0.45],
    ]
    return objects, locations


@pytest.fixture
def sample_8() -> Tuple[List[str], List[Tuple[float, float, float, float]]]:
    objects: List[str] = [
        "a cyan vase",
        "a yellow clock",
    ]
    locations: List[Tuple[float, float, float, float]] = [
        [0.1, 0.8, 0.2, 0.5],
        [0.4, 0.8, 0.3, 0.7],
    ]  # try pretrain = 'GLIGEN' if bbox not accurate
    return objects, locations


@pytest.fixture
def sample_9() -> Tuple[List[str], List[Tuple[float, float, float, float]]]:
    objects: List[str] = [
        "a yellow clock",
        "a cyan vase",
    ]
    locations: List[Tuple[float, float, float, float]] = [
        [0.4, 0.8, 0.3, 0.7],
        [0.1, 0.8, 0.2, 0.5],
    ]
    return objects, locations


@pytest.fixture
def sample_10() -> Tuple[List[str], List[Tuple[float, float, float, float]]]:
    objects: List[str] = [
        "a girl",
        "a refrigerator",
    ]
    locations: List[Tuple[float, float, float, float]] = [
        [0.2, 0.9, 0.4, 0.7],
        [0.3, 0.9, 0.4, 0.7],
    ]
    return objects, locations


@pytest.fixture
def sample_11() -> Tuple[List[str], List[Tuple[float, float, float, float]]]:
    objects: List[str] = [
        "a refrigerator",
        "a girl",
    ]
    locations: List[Tuple[float, float, float, float]] = [
        [0.2, 0.9, 0.4, 0.7],
        [0.2, 0.9, 0.3, 0.6],
    ]
    return objects, locations


@pytest.fixture
def sample_12() -> Tuple[List[str], List[Tuple[float, float, float, float]]]:
    objects: List[str] = [
        "a man wearing white shirt",
        "a cow next to a man",
        "fence",
    ]
    locations: List[Tuple[float, float, float, float]] = [
        [0.4, 1, 0, 1],
        [0.1, 1, 0.1, 0.5],
        [0.2, 1, 0.3, 0.8],
    ]
    return objects, locations


@pytest.fixture
def sample_13() -> Tuple[List[str], List[Tuple[float, float, float, float]]]:
    objects: List[str] = [
        "a cow next to a man",
        "a man wearing white shirt",
        "fence",
    ]
    locations: List[Tuple[float, float, float, float]] = [
        [0.2, 1, 0.3, 0.8],
        [0.1, 1, 0.1, 0.5],
        [0.4, 1, 0, 1],
    ]
    return objects, locations


@pytest.fixture
def sample_14() -> Tuple[List[str], List[Tuple[float, float, float, float]]]:
    objects: List[str] = [
        "park",
        "trees",
        "a fountain",
        "a lion statue",
        "bush",
    ]
    locations: List[Tuple[float, float, float, float]] = [
        [0, 1, 0, 1],
        [0, 0.7, 0, 1],
        [0, 0.7, 0.3, 0.7],
        [0.4, 0.7, 0.5, 0.8],
        [0.6, 0.9, 0.1, 0.9],
    ]
    return objects, locations


@pytest.fixture
def sample_15() -> Tuple[List[str], List[Tuple[float, float, float, float]]]:
    objects: List[str] = [
        "a brown teddy bear",
        "a computer",
    ]
    locations: List[Tuple[float, float, float, float]] = [
        [0.4, 0.8, 0.2, 0.5],
        [0.3, 0.8, 0.3, 0.8],
    ]
    return objects, locations


@pytest.fixture
def sample_16() -> Tuple[List[str], List[Tuple[float, float, float, float]]]:
    objects: List[str] = [
        "a computer",
        "a brown teddy bear",
    ]
    locations: List[Tuple[float, float, float, float]] = [
        [0.3, 0.8, 0.3, 0.8],
        [0.4, 0.8, 0.2, 0.5],
    ]
    return objects, locations


@pytest.fixture
def sample_17() -> Tuple[List[str], List[Tuple[float, float, float, float]]]:
    objects: List[str] = [
        "a piano",
        "a giant Yamaha guitar",
    ]
    locations: List[Tuple[float, float, float, float]] = [
        [0.5, 0.9, 0.2, 0.8],
        [0.1, 0.7, 0.4, 0.6],
    ]
    return objects, locations


@pytest.fixture
def sample_18() -> Tuple[List[str], List[Tuple[float, float, float, float]]]:
    objects: List[str] = [
        "a giant Yamaha guitar",
        "a piano",
    ]
    locations: List[Tuple[float, float, float, float]] = [
        [0.1, 0.7, 0.4, 0.6],
        [0.5, 0.9, 0.2, 0.8],
    ]
    return objects, locations


@pytest.fixture
def sample_19() -> Tuple[List[str], List[Tuple[float, float, float, float]]]:
    objects: List[str] = [
        "a boat",
        "a bear",
    ]
    locations: List[Tuple[float, float, float, float]] = [
        [0.4, 0.8, 0.1, 0.9],
        [0.2, 0.8, 0.3, 0.6],
    ]
    return objects, locations


@pytest.fixture
def sample_20() -> Tuple[List[str], List[Tuple[float, float, float, float]]]:
    objects: List[str] = [
        "a bear",
        "a boat",
    ]
    locations: List[Tuple[float, float, float, float]] = [
        [0.2, 0.8, 0.3, 0.6],
        [0.4, 0.8, 0.1, 0.9],
    ]
    return objects, locations


@pytest.fixture
def sample_21() -> Tuple[List[str], List[Tuple[float, float, float, float]]]:
    objects: List[str] = [
        "a girl in yellow dress next to a boy",
        "a boy in blue T-shirt",
    ]
    locations: List[Tuple[float, float, float, float]] = [
        [0.2, 1, 0.2, 0.6],
        [0.1, 1, 0.4, 0.8],
    ]
    return objects, locations


@pytest.fixture
def sample_22() -> Tuple[List[str], List[Tuple[float, float, float, float]]]:
    objects: List[str] = [
        "a boy in blue T-shirt next to a girl",
        "a girl in yellow dress",
    ]
    locations: List[Tuple[float, float, float, float]] = [
        [0.1, 1, 0.4, 0.8],
        [0.2, 1, 0.2, 0.6],
    ]
    return objects, locations


@pytest.mark.parametrize(
    argnames=("model_id", "num_inference_steps"),
    argvalues=(
        ("comin/IterComp", 25),
        ("jiuntian/gligen-xl-1024", 50),
    ),
)
@pytest.mark.parametrize(
    argnames=("objects_locations"),
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
    argnames="use_dependency_parsing",
    argvalues=(True, False),
)
def test_pipelie_stable_diffusion_xl_larender_occlusion(
    model_id: str,
    device: torch.device,
    torch_dtype: torch.dtype,
    objects_locations: Tuple[List[str], List[Tuple[float, float, float, float]]],
    num_inference_steps: int,
    use_dependency_parsing: bool,
    seed: int,
    negative_prompt: str = "",
    num_images_per_prompt: int = 1,
    width: int = 1024,
    height: int = 1024,
    alpha: float = 0.8,
):
    objects, locations = objects_locations
    assert len(objects) == len(locations)

    pipe = StableDiffusionXLLaRenderPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )
    pipe = pipe.to(device)

    gligen_options = (
        {
            "gligen_scheduled_sampling_beta": 0.4,
            "gligen_boxes": [
                # y1y2x1x2 -> x1y1x2y2
                [loc[2], loc[0], loc[3], loc[1]]
                for loc in locations
            ],
            "gligen_phrases": objects,
        }
        if "gligen" in model_id.lower()
        else {}
    )
    output = pipe(
        objects=objects,
        locations=locations,
        negative_prompt=negative_prompt,
        alpha=alpha,
        width=width,
        height=height,
        use_dependency_parsing=use_dependency_parsing,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images_per_prompt,
        generator=torch.manual_seed(seed),
        **gligen_options,  # pass GLIGEN specific options if needed
    )


@pytest.mark.parametrize(
    argnames="model_id",
    argvalues=(
        "comin/IterComp",
        "jiuntian/gligen-xl-1024",
    ),
)
def test_pipelie_stable_diffusion_xl_larender_opacity(
    model_id: str, device: torch.device, torch_dtype: torch.dtype
):
    pipe = StableDiffusionXLLaRenderPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )

    raise NotImplementedError
