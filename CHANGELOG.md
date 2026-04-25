# Changelog

All notable changes to `napari-sam3-assistant` are documented here.

## 4.0.2

### Fixed

- Improved Mask Cleanup table contrast so selected rows remain readable.
- Compacted Mask Cleanup size, hole-fill, and smoothing controls to reduce vertical scrolling.

## 4.0.1

Small compatibility release focused on SAM3.1 3D/video troubleshooting.

### Added

- Added a documented Windows `SAM3.1` workaround folder with replacement upstream `sam3` files for multiplex propagation failures caused by missing Windows kernel paths.
- Added Windows README guidance that points affected users to `windows_sam31_workaround/README.md` when `SAM3.1` multiplex propagation fails with `No available kernel. Aborting execution!`.

### Fixed

- Fixed the earlier `SAM3.1` 3D/video session-start crash caused by passing `offload_state_to_cpu` into installed `sam3` backends whose `init_state()` does not accept that keyword.
- Kept the plugin-side compatibility handling for that `start_session` / `init_state` API mismatch so propagation can begin on backends that otherwise support `SAM3.1` multiplex.

### Notes

- There are two distinct `SAM3.1` issues to separate when debugging:
  - `start_session` failure with `unexpected keyword argument 'offload_state_to_cpu'`: plugin/backend API mismatch, fixed in the plugin.
  - later `propagate_in_video` failure with `No available kernel. Aborting execution!`: runtime/kernel issue seen on some Windows systems, documented as an upstream `sam3` workaround rather than a plugin runtime change.

## 4.0.0

Major UI update for image-segmentation workflows.

### Added

- Added a persistent `Simple` / `Advanced` mode selector.
- Added `Simple` mode for common imaging tasks with fewer controls on screen.
- Added guided Simple panels for image selection, task choice, prompt setup, run actions, shared status, and compact result summary.
- Added shared task routing so Simple and Advanced use the same SAM3 execution path.
- Added shared result-visibility state so both modes can respond to preview, label, and video-session results.
- Added shared activity status for `Ready`, model loading, preview running, 3D propagation, no objects found, and task failure states.
- Added automatic image-layer refresh in Simple mode when users select or drop image layers in napari.
- Added `SAM3 Mask Operations` as a standalone napari widget for cleanup, merge, and export.
- Added a Simple `Mask Ops` run action that opens the standalone Mask Operations cleanup widget.
- Added opt-in right-click removal for clicked label values in Mask Cleanup.
- Added `Undo Last Edit` history for Mask Cleanup edits on the selected Labels layer.
- Added double-click component-table navigation in Mask Cleanup to jump the viewer to a mask centroid.
- Added an overlap-map preview action for selected class masks before final merge/export.

### Changed

- `Advanced` keeps the full manual workflow for model setup, batch work, large-image ROI controls, results tables, and logs.
- Mask Operations is no longer packed into Advanced as Step 6.
- Mask Operations opens as a standalone cleanup widget so Simple and Advanced can use the same mask-review tools.
- Simple mode is designed around a short imaging workflow: choose image/task, add the prompt, then run preview.
- Simple mode uses SAM3.0 for image tasks so Advanced SAM3.1 video-model choices do not break Simple image segmentation.
- Device selection is explicit with `GPU` and `CPU`; automatic device mode is no longer exposed in the Simple workflow.
- User-facing point-correction language now uses `Live Points` instead of `live refinement`.
- The mode labels are short: `Simple` and `Advanced`.

### Fixed

- Fixed Simple mode runs so the SAM3 image model is prepared before preview inference.
- Fixed Simple exemplar runs that could create prompt/result boxes without writing a mask until the model had been loaded manually in Advanced.
- Kept Advanced model path, model type, and device controls from overriding Simple mode's SAM3.0 image-task path.
- Kept Simple mode compact while allowing a wider, usable one-column layout.

## 3.3.0

### Fixed

- Fixed 3D/video box-only prompts so they initialize tracker propagation instead of only producing a mask on the prompted frame.
- Preserved non-empty prompted-frame masks when later 3D/video updates for the same frame are empty.
- Kept text-plus-box 3D/video prompts on the semantic box request path while routing box-only prompts through tracker box points.

## 3.2.0

### Changed

- Clarified 3D/video task guidance and documented prompt-count limits for box, point, and labels-mask prompts.
- Made the primary run button task-aware: `Run Preview` changes to `Start 3D Propagation` in 3D/video mode, and existing-session propagation is disabled until a valid session exists.

### Fixed

- Fixed 3D/video point prompting for SAM3 video predictors by sending normalized point coordinates and an object id with point prompts.
- Fixed initial 3D/video point propagation crashes caused by missing SAM3 tracker frame-cache entries before propagation.
- Fixed stale SAM3 video-session handling after cancelling or clearing prompt state so users can start a new 3D/video run without restarting napari.
- Added clear validation for unsupported 3D/video labels-mask prompts, mixed point plus text/box prompt requests, SAM3.0 multi-box video prompts, and point requests above 16 points.

## 3.1.0

Small workflow cleanup release focused on making the main widget easier to use.

### Changed

- Renamed `Step 2. Task` to `Step 2. Task Setup`.
- Removed the separate `Layers` step.
- Moved image selection into Task Setup as `Target image`.
- Added a collapsed `Advanced` section for channel axis, detection threshold, and 3D direction.
- Kept common controls visible by default: task, target image, batch mode, large-image mode, and ROI size.
- Renumbered the remaining steps so the workflow reads from setup to prompts, run, results, and mask operations.
- Kept backend behavior, prompt collection, batch processing, large-image inference, and result writing unchanged.

## 3.0.0

Major update compared with 2.0.0, focused on large-image segmentation, Step 7 mask operations, workflow modularization, and bug fixes.

### Added

- Optional large-image local inference mode for OME-Zarr, large TIFF, and similar very large images, including images on the order of `60000 x 50000` pixels when the data source can provide lazy ROI reads.
- Local ROI inference with selectable ROI sizes from `512 x 512` through `8192 x 8192`.
- Active ROI overlay layer showing the current SAM3 local inference window in global image coordinates.
- Step 7 `Mask Operations` workflow for accepting, cleaning, merging, and exporting segmentation masks.
- Accepted-object saving with object name, class name, class value, append, and replace modes.
- Class-level merge workflow for accepted object layers.
- Mask cleanup tools for connected-component analysis, deleting selected components, removing small objects, filling holes, smoothing masks, keeping the largest object, and relabeling values.
- Final merge/export tools for semantic, instance, and binary output masks.
- Overlap handling during final mask merge with priority, selection-order, component-size, and background rules.
- Mask export to TIFF, PNG, and NumPy `.npy`.
- Task-runner modules for image, refinement, and video workflows to keep the main widget easier to maintain.

### Changed

- Large-image ROI choices now include `4096 x 4096` and `8192 x 8192`.
- README now documents the large-image workflow and Step 7 mask operations.
- Result writing handles ROI-local outputs and maps them back into global image coordinates.
- Prompt collection and coordinate utilities support ROI-local conversion for large-image workflows.

### Fixed

- Improved coordinate handling for local ROI segmentation on large images.
- Improved preview/result layer handling for ROI-local outputs.
- Added tests for ROI extraction, prompt localization, global result mapping, and Step 7 UI presence.

## 2.0.0

Major update focused on SAM3.1 support, clearer model selection, live refinement, and result handling.

### Added

- SAM3.1 video multiplex support through `sam3.1_multiplex.pt`.
- Explicit `Model type` selector:
  - `SAM3.0 2D/3D/video`
  - `SAM3.1 video multiplex`
- Model-type-aware validation so SAM3.0 and SAM3.1 folders are not confused.
- SAM3.1 routing for 3D/video propagation through the multiplex video predictor.
- Automatic task guidance for SAM3.1: `Load 2D Model` is disabled and the task is set to `3D/video propagation`.
- Two-column step-based widget layout:
  - left column: `Model Setup`, `Task`, `Layers`
  - right column: `Prompt Tools`, `Run`, `Results`, and collapsible `Status`
- Muted professional widget theme for lower eye strain during long napari sessions.
- Compact activity indicator in `Run` showing model execution, propagation, refinement, idle, and failure states.
- Results table showing `Layer`, `Prompt`, `Frame`, `Object ID`, `Score`, and `Area`.
- Detection threshold control for SAM3 grounding, useful when text prompts return no candidates.
- More visible text prompt input with `Enter` bound to `Run Preview`.
- Text prompt cleanup that sends short model-facing phrases such as `myelin ring` instead of instruction text such as `segment all the myelin ring`.
- Automatic lower-threshold retry for text segmentation when the first pass returns zero objects.
- Batch mode for running the same 2D prompt setup across all open image layers, with separate preview and saved label layers per image.
- Multi-text batch mode: enter one text concept per line and run each prompt independently against the selected image or all image layers.
- Results actions:
  - `Clear Results`
  - `Copy Clipboard`
  - `Export CSV`
- Label-value merge controls for converting multiple label IDs into one class value.
- Stable object-ID label mapping for propagated video/stack results.
- Remembered model type, model directory, and device selection through Qt settings.
- Safer CUDA error reporting for unsupported GPU kernel architectures.

### Live Points Improvements

- `Live Points` mode now arms live point immediately after `Create Prompt Layer`.
- The first point starts live refinement; the first run may take longer if the model is lazy-loaded.
- `SAM3 preview labels` is pre-created for live point so napari does not switch users away from the points layer after the first result.
- After each live point update, the active layer returns to `SAM3 points` in add mode.
- `Next point mode` affects only future points.
- `T` toggles next point mode only and does not rerun refinement.
- `Shift+T` flips selected point polarity, or the latest point if none is selected, and reruns refinement.
- Existing dot colors now change only when stored point `properties["polarity"]` changes.
- `Apply mode to selected points` still edits selected existing points and reruns preview.

### Changed

- `Backend / Model Setup` was renamed to `Model Setup`.
- `Napari Layers` was renamed to `Layers`.
- `Batch all image layers` can be used independently from multi-text prompts.
- `Load Image Model` was renamed to `Load 2D Model`.
- `Lazy-load on run` was renamed to `Load model when running`.
- Text prompts no longer require a prompt layer.
- Text prompt submission can be run with `Enter` without moving to the `Run Preview` button.
- Preview clearing removes generated preview layers only and keeps prompts, saved labels, and loaded models.
- `pytorch_model.bin` was removed from documented and validated model-file names.

### Current Model Support

- SAM3.0 weights support 2D image tasks and 3D/video propagation.
- SAM3.1 `sam3.1_multiplex.pt` supports 3D/video propagation.
- SAM3.1 is not currently routed through the plugin's 2D image model loader.

## 1.0.0

Initial SAM3 Assistant plugin foundation.

### Added

- Local SAM3 backend adapter for napari workflows.
- Task-based UI for:
  - 2D segmentation
  - 3D stack/video-like propagation
  - exemplar segmentation
  - text segmentation
  - Live Points with positive and negative prompts
- Prompt collection from napari Points, Shapes, Labels, and text input.
- Box prompts from Shapes layers.
- Labels-layer mask prompts.
- Text prompts for concept segmentation.
- Preview outputs as napari Labels, Image, and Shapes layers.
- Saved label outputs through `Save Result as Labels`.
- Background worker execution to keep the napari UI responsive.
- Channel-axis handling for grayscale, RGB/RGBA, channel-first, and stack-like data.
- Basic model-directory validation for local SAM3 files.
