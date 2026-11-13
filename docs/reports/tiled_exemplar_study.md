# Tiled exemplar segmentation: reference reuse and ROI size

Study dates: November 12–13, 2026. These measurements explain the production
encoded-reference scan path and its limits. They are engineering observations
on one EM image, not an annotated biological accuracy benchmark.

## Problem and diagnosis

The legacy full scan used a natural image and localized exemplar box on tiles
containing the complete source box. Other tiles received a pasted contextual
reference beside the target. SAM3 resized the combined canvas to its model
input, changing both context and effective object scale. A fixed threshold did
not make the two prompting paths equivalent.

In `experiment_20261112_094839_57806a`, an 8192-pixel natural source tile produced
52 candidates while the pasted-reference version of that same tile produced
none. Other tiles often produced large regions rather than separate profiles.
Inspecting stitched masks before seam merging showed that the model output
already contained these regions. Seam merging sometimes compounded the problem
by joining many labels, but did not cause the original painted area.

Context ablations and native feature-reuse probes supported retaining the
exemplar's natural scene instead of pasting reference pixels into each target.
A full-resolution validation reproduced the natural source tile's saved mask
pixel for pixel with the encoded reference. Two remote 8k tiles produced 83 and
94 candidates with separated masks. These counts establish functioning feature
reuse, not correct recognition of a particular biological class.

## Production approach

1. Select a natural 8192-pixel source context containing the complete exemplar
   box. If necessary, center or expand the context to retain the full box.
   A separate crop source uses the supplied crop with a whole-crop box.
2. Encode its contextual geometry prompt once with the loaded SAM3 image model.
3. Encode each untouched target tile and reuse the captured source geometry
   features during grounding. Do not paste a source image into the target.
4. Project each accepted candidate mask at full target resolution and compose
   tile labels with existing overlap precedence. Apply seam merging separately
   where supported. Skip constant-valued tiles to avoid padding detections.

Normal full-image scans, folder batches and Z-stack scans use this path.
Folder jobs share one source encoding across files and slices; Z-stack jobs
share the selected source slice's encoding across all slices. Local ROI preview
retains its earlier behavior. Batch Experiment preserves encoded and legacy
series for comparison and can reopen earlier reports.

References belong to the adapter/model configuration that created them and are
invalid after reload, unload or configuration changes. Geometry feature reuse
uses SAM3 model internals, so backend upgrades require compatibility checks.
Uniform skipping does not eliminate false positives on textured background.

## Encoded-reference 2k versus 4k results

Saved experiment: `experiment_20261112_144824_e3c055`.
Source image: `p2mpz EM 220`, 108289 × 76326 pixels, uint8.
Both runs used threshold 0.50, overlap 15%, seam merging enabled, the same model
configuration and the same frozen source-image snapshot. All tile logs used
the same encoded source bounds: y=48741:56933, x=34815:43007.
Input target canvases were exactly 2048-square or 4096-square.

| Measurement | 2k | 4k |
|---|---:|---:|
| Total tiles | 2772 | 682 |
| Uniform tiles skipped | 611 | 119 |
| Nonuniform empty tiles | 2 | 0 |
| Raw candidates across tiles | 20790 | 13992 |
| Instance IDs before seam merging | 20441 | 13583 |
| Final instance IDs | 14051 | 8225 |
| Labeled image area | 24.79% | 43.50% |
| Largest final instance, pixels | 2244720 | 7234148 |
| Runtime, minutes | 34.2 | 21.7 |

The earlier near-full-tile blob behavior was absent in the inspected results.
Almost every empty tile was constant padding. Coverage did not change during
seam merging; merging changed identities and instance counts.

This experiment used a different exemplar from the earlier legacy experiment.
Cross-experiment totals are therefore not a controlled comparison of prompting
alone. The paired same-tile probes above provide the narrower controlled
evidence for the reference-transfer change.

## Why fixed reference features do not make ROI sizes equivalent

The source features stay fixed, but SAM3 still resizes each target crop to its
standard 1008-square image input. A structure appears approximately twice as
large in model pixels at 2k as at 4k. A 4k tile includes four times the physical
image area, more surrounding structures, and a different set of boundaries.
Feature reuse does not make the target encoder and decoder scale invariant.

Three identical 4096-square windows were examined at full output resolution:

| Source window, y/x origin | 2k coverage before merging | 4k coverage before merging |
|---|---:|---:|
| Near exemplar, 51200 / 39936 | 37.15% | 72.07% |
| Upper nerve, 45056 / 24576 | 39.67% | 73.42% |
| Lower nerve, 73728 / 32768 | 21.21% | 71.53% |

In these windows, 2k often selected smaller regions, rings or partial profiles;
4k more often filled larger bounded profiles. This observation does not assign
axon, myelin or Remak classes. Higher coverage and more instance IDs do not
establish accuracy. Colors identify labels, not biological categories.

## Evaluation and remaining work

Use Batch Experiment to hold the source, exemplar, threshold, overlap, model
and seam settings constant while sweeping ROI size. Inspect the same physical
windows before and after merging, and distinguish skipped uniform padding from
nonuniform empty detections in the tile logs. Compare intended biological units
against manual annotations for missed objects, boundary errors and split/merge
errors. Raw tile candidate counts can include overlap duplicates.

If different user-selected processing regions must produce the same model
behavior, separate those regions from a fixed, globally anchored inference
window grid. That separation is not implemented: ROI size still controls the
actual target crop. The encoded-reference change fixes a reference-transfer
failure observed here; it does not guarantee detection on every textured tile,
identical results across ROI sizes, or biological correctness.

Large-image limitations also remain: ordinary 2D scans compose a dense label
array, folder output is not a streaming huge-image pipeline, and Z-stack output
does not perform seam merging or link objects between slices. Batch Experiment
saves chunked masks and offers both composed and final stages.
