#!/usr/bin/env python3
"""Reproduce the paper's reference use, quality, and score behavior tables."""

import os
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/wmt26-task2-matplotlib")

from wmt26_mteval_task2.reference_analyses import main

if __name__ == "__main__":
    main()
