# Mask Operations

Open the mask tools from:

```text
Plugins > SAM3 Mask Operations
```

You can also click `Mask Ops` in Simple mode after a preview. The mask
operations widget works on napari `Labels` layers, including SAM3 preview layers,
saved SAM3 layers, and user-created labels.

Mask Operations is still under active development. The workflows below describe
what is useful today; labels, button placement, and specialized review tools may
continue to change faster than the core SAM3 preview workflow.

## When to Use It

Use Mask Operations after inference when you need to:

- remove small or incorrect components;
- relabel values;
- split or isolate overlapping candidate masks;
- merge several labels layers into one class or instance mask;
- inspect class overlap;
- export final TIFF, PNG, or NumPy masks.

## Tabs

| Tab | Purpose |
| --- | --- |
| `Mask Cleanup / Multiclass` | Clean one labels layer: remove small objects, relabel values, delete components, fill holes, smooth masks, and keep the largest component |
| `Mask Isolation` | Compare many aligned SAM3 mask layers and create one clean isolated object layer |
| `Merge Layers` | Combine selected labels layers into one class or instance mask |
| `Final Merge / Export` | Build final training/export masks from prepared mask layers |

`Mask Cleanup / Multiclass` also has subtabs:

| Subtab | Use it for |
| --- | --- |
| `Components` | Analyze connected components, select rows, delete components, and assign selected components to a class value |
| `Local Edit` | Use canvas-assisted local assignment or deletion tools |
| `Myelin / Axon Rings` | Experimental ring-shaped myelin/axon cleanup and proposal review |
| `Values` | Relabel, delete, keep, or convert label values |

## Basic Cleanup Workflow

1. Create or save a SAM3 preview labels layer.
2. Open `SAM3 Mask Operations`.
3. Go to `Mask Cleanup / Multiclass`.
4. Select the target labels layer.
5. Choose `Operation scope`: full layer, current slice, or a Z range.
6. On `Components`, choose a working region if the layer is very large.
7. Click `Analyze Layer`.
8. Delete unwanted components, assign selected components to a value, or apply
   cleanup actions.
9. Use `Values` when you need value-level relabeling.
10. Export or continue to merge.

The component table is intentionally rebuilt on demand. After edits that change
the mask, click `Analyze Layer` again before relying on component rows.

## Mouse-Assisted Cleanup

In `Mask Cleanup / Multiclass`, canvas tools are opt-in:

1. Open `Local Edit`.
2. Enable `Enable canvas right-click delete tools`.
3. Right-click a label object in the selected target layer to open delete
   actions.
4. Enable `Enable canvas assign tools` when you want right-click assignment to
   the current `Assignment value`.
5. Use `Undo Last Edit` if an edit was incorrect.

Double-clicking a component table row, or using locate actions in tables, jumps
the viewer to that component or candidate.

## Relabel or Merge Values

Use value tools when multiple SAM3 objects should become the same class value in
a single `Labels` layer.

Example:

```text
Values to replace: 3,4,5,6
New value: 3
```

After `Apply Relabel`, the selected labels layer is updated in place.

Other useful value actions:

- `Assign Selected To`: relabel selected values from the value table.
- `Delete Selected Values`: set selected values to background.
- `Keep Selected Only`: remove all other values.
- `Convert Non-zero To Class`: make any non-zero value the chosen class value.

## Myelin / Axon Rings

`Myelin / Axon Rings` is experimental and domain-specific. It is intended for
ring-shaped masks where the ring should remain myelin and the inner hole should
become axon.

Useful current actions:

- `Create Myelin + Axon Layers` creates separate `myelin_rings` and `axons`
  labels layers from confident proposals without changing the source labels.
- `Preview Batch Axons` creates a `batch_axon_hole_preview` layer and a proposal
  table for review.
- `Apply Confident` or `Apply Selected` edits the target labels and can be
  undone with `Undo Last Edit`.
- `Export Selected` and status-specific export buttons create proposal layers
  without editing the target.

This workflow needs both a target labels layer and a source image layer. Use the
optional ROI shape to limit proposal generation to a drawn region.

## Mask Isolation

Use `Mask Isolation` when several SAM3 candidate layers overlap and you need one
clean isolated object layer. This is useful after batch, exemplar, or repeated
prompt workflows that produce separate aligned masks.

The typical flow is:

1. Select candidate labels layers.
2. Click `Harvest Masks`.
3. Click `Classify Candidates` to mark duplicates, parents, nested children,
   fragments, and keep candidates.
4. Use the table buttons or right-click actions to keep, reject, locate, or
   isolate selected candidates.
5. Create a clean isolated labels layer.

Output options include `Create Isolated Objects Layer`, `Create Parent Layer`,
`Create Child Layer`, and `Isolate Selected`. The default included statuses are
`keep`, `nested_child`, and `parent`.

## Merge Layers

Use `Merge Layers` when prepared labels layers should be combined into one mask.
This is usually the next step after each class or object layer has been cleaned.

Decide before merging whether the output should behave like:

- a semantic mask, where each class has one label value;
- an instance mask, where each object receives a unique value;
- a binary mask, where all foreground is `1`.

In `Merge Layers`, choose whether to convert each selected layer's non-zero
pixels to one target class value, preserve source label values, or create binary
output. Selected layers must have the same shape.

## Final Merge / Export

Use `Final Merge / Export` for final deliverables. Select the prepared mask
layers, choose the output mode, inspect overlaps if needed, and export.

Buttons:

- `Merge Saved Objects`: create a final napari labels layer.
- `Show Overlap Map`: create an overlap diagnostic layer.
- `Merge and Export`: merge and write the result to disk.
- `Export Output`: export an existing output layer by name.

Overlap inspection creates:

```text
SAM3 overlap map
```

Non-zero pixels in the overlap map mark locations covered by more than one
selected class mask.

## Export Formats

Common export formats:

- TIFF for image-analysis workflows and 3D stacks;
- PNG for 2D masks where compact output is useful;
- NumPy `.npy` for Python workflows that need exact array values.

Keep a napari copy of the cleaned layer until you have verified the exported
file in the downstream tool.
