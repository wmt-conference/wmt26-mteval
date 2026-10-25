r"""Check that every config of the staged repository loads with the installed ``datasets``.

    uv run --no-project --python 3.11 --with "datasets==5.1.0" --with "numpy>=2" \
        python scripts/check_loading.py OUT 5.1.0 --grand GRAND

Use ``numpy>=2`` for 4.0.0 as well, and ``--with "datasets==2.18.0" --with "numpy<2" --with
"pyarrow<21"`` for 2.18.0 (the newest pyarrow wheels need numpy 2).  VERSION labels the output and
must be the installed version of ``datasets``.

The script links everything in OUT, plus the grand file as ``wmt26-grand.parquet``, into a
temporary directory, so that the paths in the README's YAML ``configs`` resolve as they do on the
Hub.  The links are hard links, because ``datasets`` 2.18.0 does not see symlinked files; where a
hard link is impossible (another file system) the file is copied instead.  Set TMPDIR to a
directory on the file system of OUT and GRAND to avoid copies.

Then it loads the ``evaluation`` split of every config listed in the README, the ``grand`` config
included, and prints one line each: ``OK VERSION CONFIG ROWS`` or ``FAIL VERSION CONFIG
ExceptionType: message``.  The seven data configs must all be listed in the README: one that is
not gets ``FAIL VERSION CONFIG missing from README``.  The script exits 0 if every config except
``grand`` loads (the grand file keeps its JSON columns, which older versions of ``datasets``
cannot read) and none is missing, 1 if one does not load or is missing, and 2 if the check could
not be set up.  The ``datasets`` cache lives in the temporary directory too, and
goes with it.  The script needs only ``datasets`` and PyYAML, not ``wmt26_mteval``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

GRAND_NAME = "wmt26-grand.parquet"
# The one split of every config (``wmt26_mteval.build.SPLIT``; a test checks the copy).
SPLIT = "evaluation"
# The data configs that the README must list (``grand`` is optional).  This script does not import
# wmt26_mteval, so the names are repeated here; a test checks them against ``schemas.SCHEMAS``.
REQUIRED_CONFIGS = (
    "items",
    "pairs",
    "submissions",
    "baselines",
    "systems",
    "mt_systems",
    "challenge_sets",
)
README_NAME = "README.md"
MAX_MESSAGE_CHARS = 200


def config_names(readme: str) -> list[str]:
    """The ``config_name`` of every entry of ``configs`` in the YAML front matter of ``readme``."""
    lines = readme.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{README_NAME} has no YAML front matter")
    ends = [i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---"]
    if not ends:
        raise ValueError(f"{README_NAME}: the YAML front matter is not closed")
    meta = yaml.safe_load("\n".join(lines[1 : ends[0]]))
    configs = meta.get("configs") if isinstance(meta, dict) else None
    if not configs:
        raise ValueError(f"{README_NAME}: the front matter has no configs")
    if not isinstance(configs, list) or not all(
        isinstance(config, dict) and "config_name" in config for config in configs
    ):
        raise ValueError(f"{README_NAME}: every entry of configs needs a config_name")
    return [config["config_name"] for config in configs]


def _link(source: Path, target: Path) -> None:
    """Make ``target`` a hard link to ``source``, or a copy where hard links are impossible."""
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def stage(out: Path, grand: Path, staging: Path) -> None:
    """Mirror ``out`` in ``staging`` (real directories, linked files) and add the grand file."""
    staging.mkdir(parents=True)
    for source in sorted(out.rglob("*")):
        target = staging / source.relative_to(out)
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            _link(source, target)
    _link(Path(grand), staging / GRAND_NAME)


def short_message(error: Exception) -> str:
    """The first non-empty line of the error's message, cut to ``MAX_MESSAGE_CHARS``."""
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    text = lines[0] if lines else ""
    return text if len(text) <= MAX_MESSAGE_CHARS else text[: MAX_MESSAGE_CHARS - 3] + "..."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="staged output of build_hf.py, with its README.md")
    parser.add_argument("version", help="installed version of datasets, used to label the output")
    parser.add_argument("--grand", type=Path, required=True, help="wmt26-grand.parquet")
    args = parser.parse_args(argv)

    readme = args.out / README_NAME
    try:
        names = config_names(readme.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"error: {readme}: {error}", file=sys.stderr)
        return 2

    import datasets

    if datasets.__version__ != args.version:
        print(
            f"error: datasets {datasets.__version__} is installed, not {args.version}",
            file=sys.stderr,
        )
        return 2

    required_ok = True
    with tempfile.TemporaryDirectory(prefix="check_loading_") as work:
        staging, cache = Path(work) / "staging", Path(work) / "cache"
        stage(args.out, args.grand, staging)
        for name in names:
            try:
                dataset = datasets.load_dataset(
                    str(staging), name=name, split=SPLIT, cache_dir=str(cache)
                )
                print(f"OK {args.version} {name} {len(dataset)}", flush=True)
            except Exception as error:  # any failure to load is a result to report
                print(
                    f"FAIL {args.version} {name} {type(error).__name__}: {short_message(error)}",
                    flush=True,
                )
                required_ok = required_ok and name == "grand"
    for name in REQUIRED_CONFIGS:
        if name not in names:
            print(f"FAIL {args.version} {name} missing from README", flush=True)
            required_ok = False
    return 0 if required_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
