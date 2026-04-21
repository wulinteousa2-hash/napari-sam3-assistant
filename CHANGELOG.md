# Changelog

All notable changes to `napari-sam3-assistant` are documented here.

## 3.2.0

### Changed

- Clarified 3D/video task guidance and documented prompt-count limits for box, point, and labels-mask prompts.

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

### Live Refinement Improvements

- `Refinement (live point correction)` mode now arms live point refinement immediately after `Create Prompt Layer`.
- The first point starts live refinement; the first run may take longer if the model is lazy-loaded.
- `SAM3 preview labels` is pre-created for live refinement so napari does not switch users away from the points layer after the first result.
- After each live refinement update, the active layer returns to `SAM3 points` in add mode.
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
  - refinement with positive and negative prompts
- Prompt collection from napari Points, Shapes, Labels, and text input.
- Box prompts from Shapes layers.
- Labels-layer mask prompts.
- Text prompts for concept segmentation.
- Preview outputs as napari Labels, Image, and Shapes layers.
- Saved label outputs through `Save Result as Labels`.
- Background worker execution to keep the napari UI responsive.
- Channel-axis handling for grayscale, RGB/RGBA, channel-first, and stack-like data.
- Basic model-directory validation for local SAM3 files.
