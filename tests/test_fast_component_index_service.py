from __future__ import annotations

import numpy as np

from napari_sam3_assistant.mask_operations.fast_component_index_service import (
    FastComponentIndexService,
)


def test_relabel_components_changes_only_selected_connected_component() -> None:
    data = np.zeros((6, 6), dtype=np.uint8)
    data[1:3, 1:3] = 1
    data[4:6, 4:6] = 1
    index = FastComponentIndexService().build(data)

    updated, changed = index.relabel_components(data, [1], 5)

    assert changed == 4
    assert np.all(updated[1:3, 1:3] == 5)
    assert np.all(updated[4:6, 4:6] == 1)
    assert index.record(1).label_value == 5
    assert index.record(2).label_value == 1


def test_relabel_components_keeps_component_lookup_valid() -> None:
    data = np.zeros((5, 5), dtype=np.uint8)
    data[1:4, 1:4] = 2
    index = FastComponentIndexService().build(data)

    updated, _changed = index.relabel_components(data, [1], 6)

    assert index.component_id_at((2, 2)) == 1
    assert int(updated[2, 2]) == 6
