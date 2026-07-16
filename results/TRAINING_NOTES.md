# Training notes (smoke dry-run)

## Failed attempt

First `train.py` run with `configs/data.yaml` using `path: data/smoke` failed:

```
FileNotFoundError: missing path '.../datasets/data/smoke/images/val'
```

Ultralytics resolves relative `path:` against its datasets download dir, not the repo root.

## Fix

Resolve `path` to an absolute repo path before calling `model.train()` (see `scripts/train.py` / `scripts/resolve_data_yaml.py`). Defaults stay `batch=2`, `device=cpu`, `workers=0` for ~7GB RAM laptops.
