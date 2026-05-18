# Changelog

All notable changes to `napari-sam3-assistant` are documented here.

## 4.3.3

### Fixed
- Fixed a critical napari transform crash when Simple or Advanced 2D exemplar workflows create 2D prompt/preview layers from transformed 3D image layers. Geometry copying is now dimension-aware, preserving spatial Y/X scale and translation while skipping incompatible 3D affine transforms on 2D layers.
- Added regression checks so prompt-layer geometry copying keeps fitting source image transforms to the target layer dimensionality.

## 4.3.2

### Added
- Added a `Working Region` section to `Mask Cleanup / Components` so users can analyze and clean a full-size mask by `Full mask`, `Manual ROI`, or `Drawn ROI` without indexing an entire huge label image.
- Added ROI-scoped component write-back for Mask Cleanup. Component delete, assign, and cleanup edits made inside a working ROI are written back to the original full mask at the correct coordinates.
- Added ROI-sized undo snapshots for working-region edits so local cleanup on very large masks avoids copying the full layer for each undo point.
- Added export tests for high-value `uint32` label masks.

### Changed
- Kept full-image tiled exemplar preview behavior intact after OME-Zarr fixes, while moving large-mask performance work into Mask Cleanup analysis/edit scope.
- TIFF mask export now uses `tifffile` with BigTIFF enabled so very large tiled exemplar labels and `uint32` label IDs save more reliably.

### Fixed
- Fixed Save && Clean failures on huge tiled exemplar masks where PIL TIFF export could raise errors such as `'L' format requires 0 <= number <= 4294967295`.
- Prevented PNG export from silently truncating labels above `65535`; users are directed to TIFF or NumPy for high-value label masks.

## 4.3.1

### Changed
- Copied selected image-layer geometry to SAM3 prompt layers, preview labels, and active ROI overlays so Simple and Advanced local ROI/tiled workflows align correctly on transformed OME-Zarr layers.
- Kept Simple Exemplar `Batch all image layers` independent from `Enable local/tiled inference` and kept batch Save && Clean behavior for all batch preview masks.

### Fixed
- Fixed Simple/Advanced OME-Zarr local ROI and tiled exemplar workflows where prompt boxes or points could be interpreted in the wrong coordinate system, producing empty masks while the same image worked as TIFF.
- Fixed the Mask Cleanup axon hole right-click action by keeping the inserted `QAction` object instead of overwriting it with PyQt's `None` return value.
- Fixed Save && Clean cleanup for exact tiled exemplar preview layers.

## 4.3.0

### Changed
- Refactored Simple mode into task-focused workflow tabs for 2D slice prompts, exemplar segmentation, text prompts, Live Points, 3D Multiplex, and Mask Cleanup, while keeping Advanced mode intact.
- Exposed Simple-mode controls for exemplar local/tiled inference, external crop exemplars, tile size, tile overlap, seam merging, save labels, and save-and-clean actions by reusing the existing Advanced execution paths.
- Added a compact Simple-mode activity log so task progress and save/cleanup messages are visible without switching to Advanced.
- Added compact Simple output controls for choosing the Save && Clean output folder and mask format, reusing the Advanced preview-mask export path.
- Replaced the separate Simple result summary with the Run/Output/Activity panel so users see actionable progress and save controls in one place.
- Simple `3D Multiplex` now selects the SAM3.1 model type and keeps the current viewer frame as the propagation prompt frame.
- Added a Simple `Live Points` rapid-accept workflow: right-click while the points layer is active to accept the current preview into `SAM3 live accepted labels`, optionally clear the prompt, and continue clicking the next object without manually switching layers.
- Made Simple Exemplar `Batch all image layers` independent from `Enable local/tiled inference`, and updated `Save && Clean` to export all batch preview mask layers.

### Fixed
- Fixed Mask Cleanup right-click context actions so deleting or assigning an object returns the Labels layer to `pick` mode instead of accidentally continuing a previous paint or erase tool.

## 4.2.13

### Fixed
- Fixed a napari `ValueError` when writing 2D SAM3 labels from a 3D image slice. Result layers now receive transforms that match the actual output dimensionality, so 2D labels from 3D source images keep the spatial Y/X transform without passing incompatible 3D transform values to napari.
- Fixed 2D exemplar ROI cropping from 3D stacks so the exemplar crop is taken from the current viewer frame instead of defaulting to frame 0.
- Added a regression test for 2D global-image results produced from 3D source image layers.

## 4.2.12

### Added
- Added external crop exemplar support for large-image tiled exemplar scans in Advanced mode.
- Added `Exemplar source` controls for choosing between a target-image box and a separate crop image layer.
- Added `Whole crop image` crop-region mode, which uses the full crop layer as the exemplar without requiring a Shapes box.
- Added `Box from Shapes layer` crop-region mode, which crops one boxed exemplar from the selected crop image.

### Changed
- Renamed the tiled action to `Scan Target Image by Tiles` when a separate crop image is used.
- Clarified the target/crop UI labels as `Target image to scan` and `Exemplar crop image`.
- Kept the selected target image fixed while creating prompt boxes for the external crop workflow.
- Filtered the crop-image selector so it does not offer the current target image.
- Applied the target image layer transform to global tiled exemplar outputs so masks align to the large image XY placement.
- Improved validation and logs when target image, crop image, or crop-region selection is incomplete.

## 4.2.10

### Added
- Added full-image tiled exemplar scanning for 2D Advanced exemplar segmentation.
- Added a `Scan Full Image by Tiles` action that is available when `Exemplar segmentation` and large-image local inference are enabled.
- Added a `Tile overlap` control, defaulting to `15%`, for overlap between neighboring local inference tiles.
- Added a default-on `Merge seam-split objects` option that reconnects labels cut by tile boundaries using a narrow seam-band union-find pass.
- Added composed full-size output layers named `SAM3 tiled exemplar labels`, `SAM3 tiled exemplar masks`, and `SAM3 tiled exemplar boxes`.
- Added README artwork demonstrating the new batch local exemplar segmentation workflow.
- Added a dedicated `Myelin / Axon Rings` tab inside `Mask Cleanup / Multiclass`.
- Added one-click `Create Myelin + Axon Layers` output that creates separate `myelin_rings` and `axons` Labels layers without editing the original SAM3 labels.

### Changed
- Changed the normal exemplar run button in large-image mode to `Run Current ROI Only` so users can distinguish a single local ROI test from a full-image tiled scan.
- Reused the large-image local ROI size as the tiled scan size, allowing large images such as 6k x 6k data to be segmented through smaller SAM3 calls and then composed back into original image coordinates.
- Repaired common tiled-stitch artifacts by relabeling seam-merged objects sequentially after composition.
- Improved logs and output naming for tiled exemplar scans, including tile count and labeled-pixel reporting.
- Reorganized `Mask Cleanup / Multiclass` so `Local Edit` contains only general canvas assign/delete tools.
- Moved myelin/axon proposal review buttons and the proposal table into collapsed `Advanced review`.
- Scoped the axon click tool and axon right-click action to the `Myelin / Axon Rings` tab.

## 4.2.9

### Added
- Added full-image tiled exemplar scanning for 2D Advanced exemplar segmentation.
- Added a `Scan Full Image by Tiles` action that is available when `Exemplar segmentation` and large-image local inference are enabled.
- Added a `Tile overlap` control, defaulting to `15%`, for overlap between neighboring local inference tiles.
- Added a default-on `Merge seam-split objects` option that reconnects labels cut by tile boundaries using a narrow seam-band union-find pass.
- Added composed full-size output layers named `SAM3 tiled exemplar labels`, `SAM3 tiled exemplar masks`, and `SAM3 tiled exemplar boxes`.
- Added README artwork demonstrating the new batch local exemplar segmentation workflow.

### Changed
- Changed the normal exemplar run button in large-image mode to `Run Current ROI Only` so users can distinguish a single local ROI test from a full-image tiled scan.
- Reused the large-image local ROI size as the tiled scan size, allowing large images such as 6k x 6k data to be segmented through smaller SAM3 calls and then composed back into original image coordinates.
- Repaired common tiled-stitch artifacts by relabeling seam-merged objects sequentially after composition.
- Improved logs and output naming for tiled exemplar scans, including tile count and labeled-pixel reporting.

## 4.2.8

### Added
- Added a more actionable batch axon proposal table in `Mask Cleanup / Multiclass`, including row-click locate, selected-row apply, selected-row hole filling, selected-row skip, low-confidence skip, visible-row selection, and status-based selection.
- Added sortable batch axon proposal columns so users can quickly group confident, review, failed, skipped, or custom-selected proposals.
- Added selected/status export actions for batch axon proposals so confident, review, failed, skipped, or manually selected masks can be written to separate labels layers before edits are applied.
- Added tooltips for batch axon controls and review/export actions.

### Changed
- Clarified the Mask Cleanup local-edit workflow around preview-first batch axon carving, manual review, and safe export layers.
- Kept batch axon review table cells read-only so the table is used for inspection, grouping, locating, applying, skipping, filling, and exporting rather than accidental value editing.

### Fixed
- Removed deprecated `viewer.window.qt_viewer` usage from canvas context-menu positioning.
- Fixed the batch axon review/export workflow so review groups can be separated without changing the source labels layer.

## 4.2.7

### Added
- Added a click-driven axon hole tool to `Mask Cleanup / Multiclass` for post-SAM myelin cleanup. Users can select a source image, click inside the axon region of a SAM mask, and set the seed-similar connected region to background or an axon class value.
- Added `Seed tolerance %`, `Max axon %`, and optional axon class assignment controls for tuning axon-hole cuts.
- Added visible component-analysis progress reporting in `Mask Cleanup / Multiclass` when `Analyze Layer` builds the fast component index.
- Added internal `Local Edit`, `Components`, and `Values` sections inside Mask Cleanup so local canvas edits, global component-table review, and class/value remapping have separate workflows.
- Added preview-first batch axon hole proposals in `Local Edit`, with an actionable review table, optional ROI Shapes layer filtering, a preview labels layer, and `Apply Confident` for high-confidence axon cuts.
- Added batch proposal review actions for locating selected axons, applying selected proposals, skipping selected or low-confidence proposals, and filling holes inside selected proposed axon masks.
- Added sorting, status-based selection, and export actions for batch axon proposal masks so users can create separate confident/review/failed/selected layers without editing the target labels.
- Added focused tests for axon-region extraction, clicked-label flood masks, and fast-index progress callbacks.

### Changed
- Changed axon cutting from dark-region growth to seed-similar intensity growth, so it can handle either bright axons with dark myelin rims or dark axons with bright rims.
- Made `Undo Last Edit` in Mask Cleanup return faster by restoring the previous layer data, clearing the stale component index/table, and asking users to rerun `Analyze Layer` only when they need the component table rebuilt.
- Kept axon cutting constrained to the clicked SAM label component so it does not leak into neighboring objects or background.
- Changed canvas mouse behavior to follow the active Mask Cleanup section: local flood-fill tools in `Local Edit`, fresh-index component selection/actions in `Components`, and label-value selection/actions in `Values`.
- Stopped local and value edits from silently rebuilding the global component index; edited masks now mark the component table stale until users explicitly click `Analyze Layer`.
- Batch axon preview uses a distance-transform seed inside each candidate object instead of requiring the user to click each axon manually.

### Fixed
- Fixed axon-hole clicks removing the myelin rim when the rim was darker than the axon interior.
- Fixed a `NameError` from using `Qt.WaitCursor` without importing `Qt`.
- Fixed a napari/vispy `EventEmitter loop detected` crash by removing nested `QApplication.processEvents()` calls from the canvas mouse-click path.
- Fixed slow or unclear feedback after axon-hole clicks by logging the accepted click and updating the status text without re-entering the event loop.

## 4.2.6

### Added
- Added a new `Mask Isolation` tab to Mask Operations between `Mask Cleanup / Multiclass` and `Merge Layers`.
- Added candidate consolidation for aligned SAM3 mask layers, including global candidate IDs, source provenance, duplicate detection, nested parent/child detection, fragment detection, and clean isolated output object IDs.
- Added a sortable candidate table with status, source layer, source label, component ID, area, parent/child counts, overlap metrics, and bbox metadata.
- Added visible harvest progress reporting for Mask Isolation so users can see which layer and label value is being processed.
- Added optional parent-only and child-only helper layer creation from classified candidates.
- Added concise tooltips for Mask Isolation controls, candidate statuses, context menus, and Mask Operations tabs.
- Added opt-in table right-click actions in Mask Isolation for keep, reject, duplicate, parent, child, fragment, locate, and isolate selected candidates.
- Added quick class assignment tools in `Mask Cleanup / Multiclass`, including assignment value `1` through `6`, table-row assignment, and canvas right-click assignment.
- Added focused tests for candidate consolidation and connected-component class assignment.

### Changed
- Optimized Mask Isolation harvesting with SciPy connected-component labeling and bbox-local candidate mask storage instead of full-image masks per candidate.
- Made Mask Cleanup canvas delete tools explicit and opt-in instead of enabled by default.
- Kept Mask Isolation canvas/table actions non-destructive: they change candidate status or create helper layers, but do not edit source mask pixels.
- Added `scipy` as a direct project dependency because Mask Isolation now uses SciPy for fast component harvesting.

### Fixed
- Fixed duplicate SAM3 label values across different source layers colliding during downstream consolidation by using global candidate IDs.
- Fixed uploaded mask-like image layers being invisible to Mask Isolation when they are not native napari `Labels` layers.
- Fixed confusing silent refresh behavior by showing refresh and harvest status directly in the Mask Isolation tab.
- Fixed slow harvest behavior that could make napari appear frozen on multi-layer SAM3 mask sets.

## 4.2.5

### Added
- Added fast canvas-based mask cleanup for SAM3-generated Labels layers.
- Added an in-memory component index to speed up mouse selection and deletion in large multiclass masks.
- Added direct canvas cleanup interactions: double-click or left-click to select a clicked mask/component, and right-click to open cleanup actions.
- Added faster clicked-component deletion without requiring full component re-analysis after every mouse action.

### Changed
- Simplified Mask Operations into action-first tabs: `Mask Cleanup / Multiclass`, `Merge Layers`, and `Final Merge / Export`.
- Reframed cleanup around object, value, and component actions inside a selected Labels layer instead of layer-level review states.
- Removed the Advanced Layer Review tab from the default Mask Operations interface.
- Reduced Mask Cleanup UI clutter by removing redundant mouse action/button dropdown controls.

### Improved
- Improved responsiveness for Labels layers containing many SAM3-generated masks.
- Improved right-click cleanup behavior so the context menu is tied to the clicked mask.

### Fixed
- Fixed slow interaction with large Labels layers containing many mask components.
- Fixed confusing mouse cleanup controls that exposed implementation details.
- Fixed the workflow mismatch where users reviewed layers instead of directly acting on mask components.

## 4.2.4

### Added
- Added expanded Mask Operations workflows for 2D and 3D masks.
- Added support for binary and multiclass mask cleanup workflows in Mask Operations.
- Added class-aware mask handling so users can clean and manage multiclass segmentation outputs more directly.
- Added fixed XY local ROI inference for `3D/video propagation`, including SAM3.1 multiplex, so large stacks can run video propagation on a cropped region and write results back into full-frame coordinates.

### Changed
- Updated the Mask Operations UI to better support accepted objects, class conversion, component review, and cleanup workflows.

### Fixed
- Fixed Live Points keyboard shortcuts so `T` toggles the next point mode and `Shift+T` flips the selected/latest point even when focus is on the napari canvas.
- Kept Live Points shortcuts scoped to the active Simple or Advanced mode so the two panels do not both respond to the same key press.


## 4.2.3

### Added
- Added an opt-in `Log SAM3.1 diagnostics` checkbox under `Advanced > Step 2. Task Setup > Advanced`.
- Added SAM3 video diagnostics for troubleshooting model loading, video-session startup, prompt insertion, propagation timing, and napari frame-writing performance.
- Added a dedicated diagnostics module for centralized debug/performance logging.
- Added README documentation and a screenshot showing where to enable the diagnostics option.

### Changed
- SAM3.1 video performance instrumentation is now user-controllable from the UI and remains off by default for normal runs.

### Notes
- This release is intended to support performance comparison across SAM3 Assistant, training-assistant integrations, Windows CUDA setups, and Linux/DGX-style environments without changing normal segmentation behavior.

## 4.2.2

### Changed
- On Windows, the optional completion chime now uses the native `winsound` backend first for more reliable playback, while Linux and macOS keep the standard plugin sound path with a fallback beep if needed.

## 4.2.1

### Added
- Added an optional completion chime for long-running SAM3 tasks.
- The chime plays when a preview or 3D/video propagation task completes after running for more than one minute.
- Added a UI checkbox so users can turn the completion chime on or off.

### Changed
- Long-running task completion is now easier to notice when users are working away from the screen.

## 4.2.0

### Added

- Added experimental CPU-only support for SAM3.0 2D image workflows when the environment provides a CPU-safe importable `sam3` backend such as `rhubarb-ai/sam3-cpu`.
- Documented the tested optional CPU backend as the external `rhubarb-ai/sam3-cpu` fork, which reports version `0.1.0` and is distributed under the MIT license in its own repository; it is not bundled with this plugin.
- Added [docs/cpu_only.md](docs/cpu_only.md) with a CPU-only setup path, backend verification commands, supported workflow list, BPE tokenizer notes, and ARM64/DGX Spark `decord` guidance.
- Added BPE tokenizer detection for selected model folders. The plugin now prefers `bpe_simple_vocab_16e6.txt.gz`, accepts `merges.txt.gz`, and creates `bpe_simple_vocab_16e6.txt.gz` automatically from `merges.txt` when possible.
- Added CPU-mode logging that records model type, selected runtime device, checkpoint path, BPE tokenizer path, and the experimental CPU support warning.
- Added a backend defensive error wrapper for CPU image model construction when a non-CPU-safe SAM3 backend still raises `Torch not compiled with CUDA enabled`.
- Added focused tests for device normalization, CPU-only CUDA rejection, CPU prompt support validation, CPU model-construction error wrapping, and the manual device override flag.

### Changed

- Device selection is now environment-driven for normal use. CUDA-capable PyTorch selects `GPU / CUDA`; CPU-only PyTorch selects `CPU`.
- The visible device control is now an indicator by default instead of a user preference selector.
- Manual device switching is limited to backend testing and requires `NAPARI_SAM3_ENABLE_DEVICE_OVERRIDE=1`.
- CPU mode now allows SAM3.0 2D image workflows in the plugin instead of blocking untested 2D task labels. Points, boxes, exemplar, text, and Live Points are documented as the tested CPU workflows with a CPU-safe backend.
- README guidance now keeps the main GPU/CUDA setup concise and links CPU-only users to the dedicated CPU-only setup guide.

### Fixed

- Fixed CPU-only environments accidentally inheriting a saved `cuda` setting and failing with raw `Torch not compiled with CUDA enabled` errors.
- Fixed Advanced mode startup after the CPU/GPU validation changes by importing `torch` where device availability is checked.
- Fixed Simple mode so runtime device selection follows the active environment instead of silently converting CPU back to CUDA from saved settings.

### Notes

- CPU mode is for SAM3.0 2D image workflows only. SAM3.1, 3D/video propagation, multiplex workflows, and workflows requiring the SAM3 video predictor remain CUDA/GPU-only in this plugin.
- The standard Meta `facebookresearch/sam3` backend may still allocate CUDA tensors during image model construction. CPU-only users need a CPU-safe backend installed as the importable `sam3` package.

## 4.1.0

### Added

- Added `Step 4. Run and Save` quick mask acquisition: choose output folder, format, and filename, then `Save & Clean` to save the preview mask, release temporary memory, unload the model, and open the saved folder.
- Documented tested image coverage for single-channel and RGB `2K x 2K` images, plus single-channel large-image local ROI inference around `60000 x 60000`; RGB large-image local ROI inference is noted as not yet tested.

## 4.0.4

### Fixed

- Fixed `3D/video` propagation axis handling for RGB-like and multichannel stacks so frame export, propagated labels, and box coordinates stay aligned with the selected image axes.
- Added regression tests for channel-last RGB stacks and explicit channel-axis video stacks to reduce the risk of future `3D/video` axis regressions.

## 4.0.3

### Fixed

- Fixed 2D box-only preview prompting so it uses instance box segmentation instead of grounding-style box prompting.
- Fixed 2D box-only preview masks to stay inside each prompted box, making 2D box behavior distinct from exemplar prompting.
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
