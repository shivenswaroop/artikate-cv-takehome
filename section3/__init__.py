"""Section 3 silent-bug lab.

If Artikate provides a different buggy repo, replace this package with theirs
and re-run the same find → fix → test process. This module ships a realistic
coordinate-space bug so the deliverable format is complete before their pack arrives.
"""

from section3.postprocess import boxes_from_network_output

__all__ = ["boxes_from_network_output"]
