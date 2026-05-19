# napari-sam3-assistant documentation

`napari-sam3-assistant` brings local SAM3 segmentation into napari. The plugin
has two main entry points:

- `Plugins > SAM3 Assistant`: prompt collection, SAM3 execution, previews, and
  quick save.
- `Plugins > SAM3 Mask Operations`: cleanup, isolation, merge, and export tools
  for napari `Labels` layers.

## Choose Your Path

New users should read these in order:

1. [Installation](installation.md)
2. [Model setup](model_setup.md)
3. [User guide](user_guide.md)
4. [Mask operations](mask_operations.md)
5. [Troubleshooting](troubleshooting.md)

Specialized setup:

- [CPU-only SAM3.0 setup](cpu_only.md)
- [Windows SAM3.1 workaround](../windows_sam31_workaround/README.md)

Project and maintainer notes:

- [Documentation strategy](documentation_strategy.md)
- [Changelog](../CHANGELOG.md)

## Main Concepts

`Simple` mode is the recommended starting point. It keeps the workflow short:
select an image, choose a task tab, add a prompt, and run the task.

`Advanced` mode exposes the full workflow: SAM3.0/SAM3.1 selection, model
validation, prompt layers, batch options, large-image ROI controls, 3D/video
propagation, result tables, quick save, and detailed status logs.

`Mask Operations` is separate from inference. It works on napari `Labels`
layers, including SAM3 preview layers and saved layers. This area is still
under active development, but it already covers cleanup, value relabeling,
candidate isolation, class/layer merge, overlap inspection, and export.

## Output Layers

Common preview layers:

```text
SAM3 preview labels
SAM3 preview masks
SAM3 preview boxes
SAM3 propagated preview labels
SAM3 tiled exemplar labels
SAM3 tiled exemplar masks
SAM3 tiled exemplar boxes
```

Common saved or curated layers:

```text
SAM3 saved labels
SAM3 saved propagated labels
SAM3 saved tiled exemplar labels
isolated_objects
merged_class_mask
final_training_mask
```

Batch workflows append the source image name or text prompt to the layer name so
separate results remain identifiable.
