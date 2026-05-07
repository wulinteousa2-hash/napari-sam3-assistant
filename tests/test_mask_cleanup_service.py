from __future__ import annotations

import numpy as np
import pytest

from napari_sam3_assistant.mask_operations.cleanup_service import MaskCleanupService


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
