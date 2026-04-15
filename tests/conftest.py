import pytest
import tempfile
import os
from library.tool_library import ToolLibrary
import numpy as np


@pytest.fixture
def tmp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_tools.db")
        lib = ToolLibrary(db_path=db_path, seed=False)
        yield lib


@pytest.fixture
def fake_embedding():
    rng = np.random.default_rng(42)
    vec = rng.random(384).astype(np.float32)
    return vec / np.linalg.norm(vec)
