"""tests/conftest.py - 共享测试配置与 fixtures"""

import pytest

from nano_vllm.engine.sequence import Sequence


@pytest.fixture(autouse=True)
def reset_block_size():
    """确保每个测试使用默认 block_size，避免跨测试污染"""
    original = Sequence.block_size
    yield
    Sequence.block_size = original
