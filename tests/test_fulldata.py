"""Tests that need the real release: ``OUT=… GRAND=… uv run pytest -q -m fulldata``.

``OUT`` is a build of ``GRAND`` made with ``scripts/build_hf.py``.  Both are read only.
"""

import pytest

from wmt26_mteval.validate import validate


@pytest.mark.fulldata
def test_full_release(full_out, full_grand, expected_full):  # needs OUT built from GRAND
    assert validate(full_out, full_grand, expected_full) == []
