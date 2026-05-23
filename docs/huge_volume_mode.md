# Huge Volume Mode

Huge Volume Mode is the workflow for datasets too large to materialize as one
NumPy image or Labels layer, such as `400 x 70000 x 40000` EM volumes. The core
rule is that inference, mask writing, and later cleanup must operate on small
chunks and write results back to disk immediately.

## Current Phase 1 Implementation

The first implementation is available for Advanced Exemplar workflows. Enable
`Enable large-image local inference`, check `Scan all Z slices to OME-Zarr`, and
click `Scan Z Stack by Tiles`. The plugin repeats the 2D tiled exemplar scan for
each Z/frame slice and writes each tile directly into `s0` of an output mask
OME-Zarr store. It does not create a dense full-volume `Labels` array.

Current limits:

- input can be an OME-Zarr or 3D TIFF layer if napari/tifffile can read slice
  tiles lazily enough;
- output is OME-Zarr only;
- seam merging is skipped in this direct-write path;
- Mask Operations chunk write-back is still a later phase;
- object IDs are unique within each Z slice, not reconciled across Z yet.

## Target Data

Supported source inputs should include:

- OME-Zarr image groups, preferred for native chunked access;
- large TIFF or BigTIFF sources, opened lazily when possible;
- TIFF stacks converted to OME-Zarr before long batch processing when lazy TIFF
  access is not reliable enough.

The preferred working layout is OME-Zarr for both image and mask data. A typical
chunk shape is:

```text
image: (1, 1024, 1024) or (1, 2048, 2048)
mask:  (1, 1024, 1024) or (1, 2048, 2048)
```

For a `400 x 70000 x 40000` uint16 mask, a full in-memory array would be around
`2.24 TB` before overhead. The plugin must not create that array.

## Current User Workflow

1. Open a 3D image layer, preferably OME-Zarr. A 3D TIFF can work when napari and
   tifffile can read slices and tiles lazily enough.
2. Set `Task` to `Exemplar segmentation`.
3. Enable `Enable large-image local inference`.
4. Choose the tile size with `ROI size` and set `Tile overlap`.
5. Draw or select an exemplar region.
6. Check `Scan all Z slices to OME-Zarr`.
7. Click `Scan Z Stack by Tiles` and choose an output folder if needed.
8. The plugin creates an output `.ome.zarr` mask store and writes tile labels to
   `s0[z, y0:y1, x0:x1]`.

Planned workflow extensions include selectable Z ranges, resumable jobs, and
Mask Operations chunk/ROI write-back.

## Architecture Changes

### 1. Source Abstraction

Add a source adapter layer that hides whether input came from OME-Zarr, TIFF, or
a napari lazy layer.

Responsibilities:

- expose canonical shape as `(z, y, x)` or `(y, x)`;
- expose axis names and physical scale where available;
- provide `read_chunk(z_slice, y_slice, x_slice) -> np.ndarray`;
- report chunking, dtype, and source metadata;
- avoid `np.asarray(full_data)` on huge sources.

Suggested module:

```text
src/napari_sam3_assistant/huge_volume/sources.py
```

### 2. Mask Store Abstraction

Add an output mask writer backed by OME-Zarr.

Responsibilities:

- create a label OME-Zarr store with matching axes, shape, chunks, dtype, and
  scale metadata;
- write label chunks by `(z, y, x)` region;
- read chunk/ROI masks for review;
- keep a small metadata table for processed chunks, failures, prompts, model,
  and parameters.

Suggested module:

```text
src/napari_sam3_assistant/huge_volume/mask_store.py
```

### 3. Chunk Scheduler

Add a scheduler that builds chunk jobs without loading image data.

Responsibilities:

- generate tile bounds over selected `z`, `y`, and `x` ranges;
- support overlap and edge clipping;
- skip chunks that are already complete unless rerun is requested;
- persist job state so long runs can resume;
- provide progress events to the UI.

Suggested module:

```text
src/napari_sam3_assistant/huge_volume/scheduler.py
```

### 4. SAM3 Chunk Runner

Add a runner that processes one chunk at a time.

Responsibilities:

- read one image chunk;
- normalize/prepare it exactly like existing ROI inference;
- run SAM3 on a 2D plane or small selected slab;
- translate local mask coordinates back to global `(z, y, x)`;
- write only the resulting chunk/ROI to the OME-Zarr mask store;
- release model/intermediate memory between chunks when configured.

Initial scope should be 2D slice tiles. Full 3D propagation across hundreds of
huge slices should be a later phase because it needs explicit memory and session
state limits.

Suggested module:

```text
src/napari_sam3_assistant/huge_volume/runner.py
```

### 5. Napari Layer Strategy

Huge Volume Mode should not add a full dense in-memory Labels layer. Instead:

- show only the active image chunk as a preview layer;
- show only the active mask chunk or ROI overlay;
- refresh the visible chunk after writes;
- optionally expose the output OME-Zarr as a lazy Labels layer if napari can keep
  it lazy end-to-end.

The existing `SAM3 preview labels` and `SAM3 tiled exemplar labels` layers are
still useful for normal 2D work, but they are not the right storage model for a
multi-terabyte volume.

### 6. Mask Operations Integration

Mask Operations needs a chunk/ROI backend before it can safely edit huge volumes.

Required changes:

- add a `Huge Volume` working region type backed by the mask store;
- run component analysis only inside the current chunk/ROI;
- write edits back to the same OME-Zarr region;
- store undo snapshots per chunk/ROI, not per full volume;
- prevent global operations from calling `np.asarray()` on the full mask;
- label UI actions clearly when an operation is local-only.

### 7. TIFF Handling

TIFF can be accepted at the beginning, but the safest long-run path is:

1. inspect TIFF shape, dtype, axes, and page layout;
2. if the volume is huge, recommend or offer conversion to OME-Zarr;
3. process the OME-Zarr working copy;
4. export TIFF only for small final ROIs or selected slices, not the full huge
   mask volume.

A direct lazy TIFF adapter can be useful for previews and migration, but OME-Zarr
should be the canonical processing format.

## Implementation Phases

### Phase 1: 2D Z-Stack Tiled Segmentation

- Added a UI entry point for Z-stack tiled exemplar scanning.
- Added direct OME-Zarr chunk write-back for tiled labels.
- Added tests with small synthetic arrays that mimic huge-volume chunk behavior.

Remaining Phase 1 hardening:

- add resumable job metadata;
- add source inspection logs: shape, dtype, chunks, axes, scale, estimated mask
  size;
- add optional Z range controls instead of always scanning all slices.

### Phase 2: Chunk Mask Operations

- Add Mask Operations read/write-back for one chunk/ROI.
- Add local component cleanup, relabeling, and export for selected chunks.
- Add per-chunk undo and stale-analysis handling.

### Phase 3: Advanced Volume Workflows

- Add optional adjacent-slice propagation where memory allows.
- Add chunk-boundary reconciliation for objects split across tiles or slices.
- Add batch QA summaries and selected-region export.

## Non-Goals For The First Version

- No full-volume connected-component analysis.
- No full-volume dense Labels layer in memory.
- No single full-volume TIFF mask export for terabyte-scale masks.
- No assumption that SAM3 can process a 400-slice volume as one session.

## Acceptance Criteria

Current Phase 1 is acceptable when it can:

- reference a 3D image layer without converting it to a full NumPy volume;
- create an output mask OME-Zarr store;
- run tiled 2D exemplar inference through all Z/frame slices;
- write each tile to the correct global `z, y, x` region;
- reopen the output mask and verify written chunks align with the source.

The broader Huge Volume roadmap is complete only after resumable jobs, source
inspection, Z range controls, chunk/ROI Mask Operations write-back, seam
reconciliation, and cross-Z object linking are implemented.
