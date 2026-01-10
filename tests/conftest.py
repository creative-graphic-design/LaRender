import pytest
import torch


@pytest.fixture
def seed() -> int:
    return 42


@pytest.fixture
def device() -> torch.device:
    # return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("mps")


@pytest.fixture
def torch_dtype() -> torch.dtype:
    return torch.float16
