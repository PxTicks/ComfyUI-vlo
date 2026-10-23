"""Print the core-source fingerprints the pack's copies of core code are pinned to.

    COMFYUI_PATH=~/ComfyUI python tests/regen_fingerprint.py

Only paste a result into its module after reviewing the upstream diff and
re-running that module's tests -- the fingerprint is what stops stale copied code
running against a newer core, so updating it blindly defeats it:

- masked guides (`nodes/minimax_masked_guide/compatibility.py`): review
  `comfy/ldm/minimax/model.py`, re-run
  `tests/test_masked_guide_forward_equivalence.py`;
- SAM3 refine (`nodes/sam3_refine.py`): review `comfy/ldm/sam3`, re-run
  `tests/test_sam3_refine.py` with `VLO_SAM3_CHECKPOINT` set.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from minimax_h3_harness import masked_guide_module  # noqa: E402
from test_batch_nodes_integration import _load_nodes_module  # noqa: E402


def _report(name, module) -> bool:
    found = module.core_source_fingerprint()
    expected = module.TESTED_SOURCE_FINGERPRINT
    print("{}:".format(name))
    print("  expected (pinned): {}".format(expected))
    print("  found (installed): {}".format(found))
    print("  " + ("MATCH" if found == expected else "MISMATCH - review the upstream diff first"))
    return found == expected


def main() -> int:
    results = [
        _report("masked guides", masked_guide_module("compatibility")),
        _report("SAM3 refine", importlib.import_module(_load_nodes_module().__name__ + ".sam3_refine")),
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
