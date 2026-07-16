# Section 3 — Silent bug lab

Self-contained detector postprocess with a historical silent coordinate bug (wrong letterbox `pad_h` sign). See `ANSWERS.md` Section 3.

- Bug lived in `postprocess.py` (`+= pad_h` instead of `-= pad_h`)
- Regression: `tests/test_section3_regression.py`

If Artikate provides a separate buggy repo, drop it here (or alongside) and repeat find → fix → test; update `ANSWERS.md` with that file/line.