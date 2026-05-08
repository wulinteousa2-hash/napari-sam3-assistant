from __future__ import annotations

import numpy as np
import pytest

from napari_sam3_assistant.mask_operations.cleanup_service import MaskCleanupService
from napari_sam3_assistant.mask_operations.models import AxonHoleCandidate


def test_connected_label_mask_floods_only_clicked_component() -> None:
    labels = np.zeros((5, 7), dtype=np.uint8)
    labels[1:3, 1:3] = 1
    labels[1:3, 5:7] = 1

    mask = MaskCleanupService().connected_label_mask(labels, (1, 1), 1)

    assert int(np.count_nonzero(mask)) == 4
    assert mask[1, 1]
    assert not mask[1, 5]


def test_cut_seed_similar_region_erases_dark_inner_axon_only() -> None:
    labels = np.zeros((9, 9), dtype=np.uint8)
    labels[1:8, 1:8] = 1
    image = np.full((9, 9), 120, dtype=np.uint8)
    image[3:6, 3:6] = 20
    component = labels == 1

    updated, changed, axon_pixels = MaskCleanupService().cut_seed_similar_region(
        labels,
        image,
        component,
        (4, 4),
        output_value=0,
        threshold_percent=25,
        max_fraction_percent=75,
    )

    assert changed == 9
    assert axon_pixels == 9
    assert np.all(updated[3:6, 3:6] == 0)
    assert updated[1, 1] == 1


def test_cut_seed_similar_region_can_assign_axon_class() -> None:
    labels = np.zeros((7, 7), dtype=np.uint8)
    labels[1:6, 1:6] = 1
    image = np.full((7, 7), 100, dtype=np.uint8)
    image[3, 3] = 10

    updated, changed, axon_pixels = MaskCleanupService().cut_seed_similar_region(
        labels,
        image,
        labels == 1,
        (3, 3),
        output_value=2,
        threshold_percent=20,
        max_fraction_percent=75,
    )

    assert changed == 1
    assert axon_pixels == 1
    assert int(updated[3, 3]) == 2
    assert int(updated[2, 3]) == 1


def test_cut_seed_similar_region_keeps_bright_axon_when_ring_is_dark() -> None:
    labels = np.zeros((9, 9), dtype=np.uint8)
    labels[1:8, 1:8] = 1
    image = np.full((9, 9), 120, dtype=np.uint8)
    image[2:7, 2:7] = 40
    image[3:6, 3:6] = 105

    updated, changed, axon_pixels = MaskCleanupService().cut_seed_similar_region(
        labels,
        image,
        labels == 1,
        (4, 4),
        output_value=0,
        threshold_percent=20,
        max_fraction_percent=75,
    )

    assert changed == 9
    assert axon_pixels == 9
    assert np.all(updated[3:6, 3:6] == 0)
    assert updated[2, 4] == 1


def test_cut_seed_similar_region_rejects_region_above_max_fraction() -> None:
    labels = np.ones((6, 6), dtype=np.uint8)
    image = np.full((6, 6), 20, dtype=np.uint8)

    with pytest.raises(ValueError, match="above the 20% limit"):
        MaskCleanupService().cut_seed_similar_region(
            labels,
            image,
            labels == 1,
            (3, 3),
            output_value=0,
            threshold_percent=10,
            max_fraction_percent=20,
        )


def test_propose_axon_holes_uses_distance_seed_and_preview_mask() -> None:
    labels = np.zeros((12, 12), dtype=np.uint8)
    labels[2:10, 2:10] = 1
    image = np.full((12, 12), 120, dtype=np.uint8)
    image[4:8, 4:8] = 30

    candidates, masks, preview = MaskCleanupService().propose_axon_holes(
        labels,
        image,
        threshold_percent=25,
        max_fraction_percent=75,
        min_object_size=8,
        min_confidence_percent=70,
    )

    assert len(candidates) == 1
    assert candidates[0].status == "confident"
    assert candidates[0].axon_area == 16
    assert int(np.count_nonzero(masks[candidates[0].candidate_id])) == 16
    assert int(np.count_nonzero(preview)) == 16


def test_apply_axon_hole_candidates_updates_only_confident_masks() -> None:
    labels = np.zeros((12, 12), dtype=np.uint8)
    labels[2:10, 2:10] = 1
    image = np.full((12, 12), 120, dtype=np.uint8)
    image[4:8, 4:8] = 30
    service = MaskCleanupService()
    candidates, masks, _preview = service.propose_axon_holes(
        labels,
        image,
        threshold_percent=25,
        max_fraction_percent=75,
        min_object_size=8,
        min_confidence_percent=70,
    )

    updated, applied, changed = service.apply_axon_hole_candidates(
        labels,
        candidates,
        masks,
        output_value=2,
        min_confidence_percent=70,
    )

    assert applied == 1
    assert changed == 16
    assert np.all(updated[4:8, 4:8] == 2)
    assert updated[2, 2] == 1


def test_fill_axon_candidate_holes_updates_selected_proposal() -> None:
    service = MaskCleanupService()
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False
    candidates = [
        AxonHoleCandidate(
            candidate_id=1,
            label_value=1,
            object_area=64,
            axon_area=24,
            axon_fraction=24 / 64,
            confidence=0.7,
            status="confident",
            reason="passes area and boundary checks",
            seed=(4, 4),
            bbox=((2, 7), (2, 7)),
        )
    ]
    masks = {1: mask}
    before = int(np.count_nonzero(masks[1]))

    updated_candidates, updated_masks, filled = service.fill_axon_candidate_holes(
        candidates,
        masks,
        {1},
        min_confidence_percent=70,
        max_fraction_percent=75,
    )

    assert filled == 1
    assert int(np.count_nonzero(updated_masks[1])) == before + 1
    assert updated_candidates[0].reason == "filled holes in axon proposal"
