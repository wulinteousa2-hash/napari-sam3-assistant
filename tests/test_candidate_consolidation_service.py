from __future__ import annotations

import numpy as np
import pytest

from napari_sam3_assistant.mask_operations.candidate_consolidation_service import (
    CandidateConsolidationService,
)


class FakeLayer:
    def __init__(self, name: str, data: np.ndarray) -> None:
        self.name = name
        self.data = data
        self.visible = True


class FakeLayers(list):
    def __getitem__(self, key):
        if isinstance(key, str):
            for layer in self:
                if layer.name == key:
                    return layer
            raise KeyError(key)
        return super().__getitem__(key)


class FakeViewer:
    def __init__(self, layers: list[FakeLayer]) -> None:
        self.layers = FakeLayers(layers)


def test_harvest_assigns_global_candidate_ids_for_duplicate_source_values() -> None:
    layer_a = np.zeros((8, 8), dtype=np.uint8)
    layer_a[1:3, 1:3] = 1
    layer_a[5:7, 5:7] = 2
    layer_b = np.zeros((8, 8), dtype=np.uint8)
    layer_b[1:3, 5:7] = 1
    layer_b[5:7, 1:3] = 2
    service = CandidateConsolidationService(
        FakeViewer([FakeLayer("Layer_A", layer_a), FakeLayer("Layer_B", layer_b)])
    )

    records = service.harvest_layers(["Layer_A", "Layer_B"], min_area=1)

    assert [record.candidate_id for record in records] == [1, 2, 3, 4]
    assert [
        (record.source_layer_name, record.source_label_value, record.source_component_id)
        for record in records
    ] == [
        ("Layer_A", 1, 1),
        ("Layer_A", 2, 2),
        ("Layer_B", 1, 1),
        ("Layer_B", 2, 2),
    ]


def test_nested_parent_and_child_are_preserved_with_parent_trimmed() -> None:
    parent = np.zeros((8, 8), dtype=np.uint8)
    parent[1:7, 1:7] = 1
    child = np.zeros((8, 8), dtype=np.uint8)
    child[3:5, 3:5] = 1
    service = CandidateConsolidationService(
        FakeViewer([FakeLayer("Parent", parent), FakeLayer("Child", child)])
    )

    records = service.harvest_layers(["Parent", "Child"], min_area=1)
    service.classify_candidates(nested_containment=0.90)
    data, mapping = service.create_isolated_label_data(
        {"keep", "nested_child", "parent"},
        overlap_rule="small_objects_win",
        parent_handling="trim_parent",
    )

    statuses = {record.source_layer_name: record.status for record in records}
    assert statuses == {"Parent": "parent", "Child": "nested_child"}
    assert mapping == {2: 1, 1: 2}
    assert np.all(data[3:5, 3:5] == 1)
    assert data[1, 1] == 2


def test_shape_mismatch_is_rejected_without_partial_records() -> None:
    service = CandidateConsolidationService(
        FakeViewer(
            [
                FakeLayer("A", np.zeros((4, 4), dtype=np.uint8)),
                FakeLayer("B", np.zeros((5, 4), dtype=np.uint8)),
            ]
        )
    )

    with pytest.raises(ValueError, match="Layer shape mismatch"):
        service.harvest_layers(["A", "B"], min_area=1)

    assert service.records == []
    assert service.candidate_masks == {}
