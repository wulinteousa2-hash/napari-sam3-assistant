# Batch experiments

In **Advanced** mode, click **Batch Experiment** to open or inspect a saved
experiment at any time. **Open experiment…** does not require an image,
exemplar, model configuration or loaded model.

To start a new batch, first choose **Exemplar segmentation**, enable tiled
inference, and select one target image and an exemplar. Validation and capture
of these inputs happen when you click **Start batch**, not when opening the
dialog. Legacy series require the ROI size to fit the exemplar. Encoded-only
series can use smaller target tiles because no reference pixels are pasted.

Select the ROI sizes to compare (2048, 4096 and 8192 are selected initially) and
an output folder. Source validation updates ROI eligibility for legacy series.
The dialog shows the number of runs and tile inferences before starting.

Choose a prompting series:

- **Encoded native reference — standard tile scan:** encode the box once in its
  natural 8k source scene, then reuse those contextual features on every original
  target tile. Source context stays fixed when ROI size changes. This is the new
  standard for all full-image scans, folder batches, Z-stack scans and experiments.
- **Compare encoded reference and legacy current:** compare the new path with
  the previous behavior as separate series.
- **Legacy current: natural / pasted:** a tile containing the complete original box
  uses the natural image and localized box; other tiles use a pasted reference.
- **Legacy pasted-reference only:** every tile uses a pasted reference,
  including tiles containing the original box. The original pixels in the
  target tile remain present, so this can duplicate the exemplar in the input.
- **Both legacy modes:** run two separate ROI-size series. Only ROI size changes
  within a series; prompting behavior is the factor between series.

The image plane is copied once to the output folder. The exemplar pixels,
coordinates, detection threshold, overlap, model configuration and seam-merge
setting are captured for the whole batch. The selected frame/channel is fixed;
this feature does not batch image layers, folders or Z stacks.

Encoded references are created fresh for each normal scan or experiment. They
are invalid after model unload, reload or configuration changes. The target
image is never augmented with reference pixels in encoded mode. Masks are
projected at full target resolution one candidate at a time; this is not the
reduced-resolution research preview. Uniform image tiles are skipped and
recorded separately to prevent predictions on constant padding. Nonuniform
background can still produce false positives.

For a box from the target image, encoding uses a full natural 8192-pixel source
tile containing the complete box, independent of target ROI size. If the box
straddles that grid, a centered source context retains it completely. Separate
crop-image sources are encoded as their own supplied image with a whole-crop
box; they have only the context present in that crop.

**Run Current ROI Only** retains the original preview behavior. All production
full-scan workflows use encoded reuse without a legacy fallback. Folder batches
encode the selected source once for every file and slice; Z-stack scans encode
the selected source slice once for the whole stack. Legacy full-scan modes are
available here for comparison and older saved experiments remain readable.

Runs are sequential. Each run saves a composed OME-Zarr label mask before seam
merging and, when merging is enabled, a separate final OME-Zarr label mask.
The engine writes chunks rather than retaining full masks for every run in RAM.
Large source images and multiple masks still need substantial disk space.

Enable **Also save decoded per-tile predictions** to keep each target tile's
labels, boxes and scores before full-image composition. These are downstream of
the adapter's mask-to-label conversion: overlapping individual candidate masks
are not saved independently. This option increases storage usage.

## Reviewing results

The comparison table shows runtime, **Objects before**, **Objects after**, final
coverage and largest final-mask area. Select a completed run and click
**Inspect Before** or **Inspect After**. Before is already stitched into a
full-image mask; it excludes only the object-joining seam-merge step. The selected
row's explanation shows how the object count changed and whether painted area
stayed the same. Results use the source image transform and the same viewer
position and zoom. Previously loaded experiment results are hidden when
another is loaded, so you can compare the same physical region by switching
between runs or toggling their layers.

**Open results folder** provides access to:

- `experiment.json`: captured settings, provenance, run states and statistics;
- `comparison.csv`: per-run measurements, tile counts, empty tiles, natural and
  reference path counts, and seam-merge groups;
- `exemplar.npy` and `source.zarr`: the fixed exemplar and source image plane;
- each run's `tiles.jsonl`: tile coordinates, prompting path, input dimensions,
  prompt coordinates, target boxes, scores, timing and processed-empty status;
- each run's masks and optional `tiles/*.npz` decoded predictions.

Use **Open experiment…** to reopen a saved `experiment.json` later. Keep the
experiment folder together so its relative mask paths continue to work.

**Cancel batch** stops at the next cancellation check, including between tiles
and output chunks. It does not interrupt an active model forward pass. Completed
runs and partial logs remain on disk; incomplete runs are marked cancelled.
An inference error marks the current run failed and ends the batch instead of
counting the failed tile as an empty detection.

## Interpreting the experiment

Coverage and number of colored labels do not measure biological accuracy. Review
whether axons, myelin and Remak bundles are correctly isolated. Label colors are
instance identities, not semantic classes.

Changing ROI size also changes target context, resizing and tile boundaries.
In the legacy current series it also changes the natural-versus-pasted path
distribution. Encoded reuse removes that path-selection switch and holds the
source features fixed, but does not keep effective target object resolution
fixed. The legacy pasted-reference series also removes the switch, but still
combines source and target pixels into one resized input.
Compare before/after merging separately from model quality, and compare the same
physical regions across results. A small manually annotated evaluation set is
needed for precision/recall and split/merge accuracy.

See the [tiled exemplar study](reports/tiled_exemplar_study.md) for measured
2k/4k results, the evidence behind encoded reuse, and remaining limitations.
