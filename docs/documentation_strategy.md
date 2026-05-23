# Documentation Strategy

This project is still young, so the documentation should describe stable user
tasks rather than overfitting every paragraph to the current widget layout.

## Goals

- Help a first-time user install the plugin, find the widget, load a model, and
  produce a preview.
- Explain the difference between Simple mode, Advanced mode, and Mask
  Operations.
- Keep advanced workflows discoverable without making the README too large.
- Document known limits honestly, especially CUDA, SAM3.1, CPU-only, and 3D/video
  constraints.
- Keep screenshots useful but not required for every workflow step.

## Structure

The README is the project front door. It should stay short and answer:

- what the plugin does;
- who it is for;
- what is required;
- where the full docs live.

The `docs/` folder carries task documentation:

- `installation.md`: environment setup and verification;
- `model_setup.md`: model files, folder layout, device rules;
- `user_guide.md`: Simple, Advanced, prompts, large images, batch, 3D/video;
- `mask_operations.md`: cleanup, isolation, merge, export;
- `troubleshooting.md`: common failures and concrete checks;
- `cpu_only.md`: specialized CPU-only setup;
- `huge_volume_mode.md`: architecture task for chunk-based TIFF/OME-Zarr volume segmentation.

## Writing Rules

- Prefer task names over implementation details.
- Keep steps numbered when the user is expected to perform them in order.
- Use exact UI labels in backticks.
- Mention limitations near the workflow they affect.
- Avoid promising stable UI layout while the plugin is actively changing.
- Update screenshots only when they would prevent confusion.
- Move release history to `CHANGELOG.md`; do not let the README become the
  changelog.

## Maintenance Checklist

Before each release:

1. Check that `README.md` links still work.
2. Confirm the main menu labels in `src/napari_sam3_assistant/napari.yaml`.
3. Confirm task names and button labels in Simple and Advanced mode.
4. Confirm Mask Operations tab names.
5. Update known limitations in `troubleshooting.md`.
6. Add detailed release notes to `CHANGELOG.md`, not the README.

## Screenshot Policy

Use screenshots for orientation, not as the only source of truth. A screenshot
should show where a user is in the app, while the written steps explain what to
do. If the UI changes often, keep screenshots broad enough that they remain
useful across small layout changes.
