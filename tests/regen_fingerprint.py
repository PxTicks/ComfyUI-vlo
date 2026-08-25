"""Print the core-source fingerprint the masked-guide fork should be pinned to.

    COMFYUI_PATH=~/ComfyUI python tests/regen_fingerprint.py

Only paste the result into `compatibility.py` after reviewing the upstream diff
to `comfy/ldm/minimax/model.py` and re-running
`tests/test_masked_guide_forward_equivalence.py` -- the fingerprint is what stops
a stale fork running against a newer core, so updating it blindly defeats it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from minimax_h3_harness import masked_guide_module  # noqa: E402


def main() -> int:
    compatibility = masked_guide_module("compatibility")
    found = compatibility.core_source_fingerprint()
    expected = compatibility.TESTED_SOURCE_FINGERPRINT
    print("expected (pinned): {}".format(expected))
    print("found (installed): {}".format(found))
    print("MATCH" if found == expected else "MISMATCH - review the upstream diff first")
    return 0 if found == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
