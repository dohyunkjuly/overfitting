"""The README usage example must run as written, from the repository root."""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def readme_example() -> str:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
    assert blocks, "README.md has no python example"
    return blocks[0]


def test_readme_example_reads_bundled_data():
    code = readme_example()
    paths = re.findall(r"read_csv\('([^']+)'\)", code)

    assert paths, "README example no longer loads a CSV"
    for path in paths:
        assert (ROOT / path).is_file(), f"README example points at a missing file: {path}"


@pytest.mark.slow
def test_readme_example_runs():
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", readme_example()],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert "Performance Summary" in result.stdout
    assert "Total Trades" in result.stdout
