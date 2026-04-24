Windows SAM3.1 workaround files

These are replacement files for the upstream `sam3` package intended for
Windows troubleshooting of SAM3.1 multiplex propagation.

Files included:

- `sam3/model/decoder.py`
  - Stops forcing Flash-only SDPA on Windows.
  - Allows fallback to `EFFICIENT_ATTENTION` or `MATH`.
  - Optional env override: `SAM3_FORCE_SDPA_FALLBACK=1`

- `sam3/model/sam3_tracker_utils.py`
  - Falls back from Triton EDT to the slow OpenCV path when the Triton kernel
    is unavailable.
  - On Windows it prefers the slow EDT path by default.
  - Optional env override: `SAM3_FORCE_SLOW_EDT=1`

- `sam3/perflib/__init__.py`
  - Disables perflib by default on Windows.
  - You can still re-enable it explicitly with `USE_PERFLIB=1`.

Suggested Windows test order:

1. Replace these three files in your installed `sam3` package.
2. Test SAM3.1 multiplex propagation with:
   - `USE_PERFLIB=0`
   - `SAM3_FORCE_SDPA_FALLBACK=1`
3. If needed also set:
   - `SAM3_FORCE_SLOW_EDT=1`

Package location:

Find the installed package with:

```python
import sam3
print(sam3.__file__)
```

Then replace the matching files under that package root.

This is a troubleshooting workaround, not a confirmed upstream fix.
