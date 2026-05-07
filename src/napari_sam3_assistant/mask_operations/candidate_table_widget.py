from __future__ import annotations

from collections.abc import Callable

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QMenu,
    QTableWidget,
    QTableWidgetItem,
)

from .component_table_widget import NumericTableWidgetItem
from .models import CandidateObjectRecord


STATUS_TOOLTIPS = {
    "keep": "Included in final isolated layer.",
    "reject": "Excluded from final isolated layer.",
    "duplicate": "Excluded by default because another candidate represents it.",
    "parent": "Included by default as a larger parent object.",
    "nested_child": "Included by default as a smaller child object.",
    "fragment": "Excluded by default as a partial fragment.",
}


class CandidateTableWidget(QTableWidget):
    def __init__(
        self,
        locate_callback: Callable[[int], None] | None = None,
        action_callback: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(0, 11)
        self._locate_callback = locate_callback
        self._action_callback = action_callback
        self._actions_enabled = True
        self.setObjectName("candidateIsolationTable")
        self.setHorizontalHeaderLabels(
            [
                "ID",
                "Status",
                "Source Layer",
                "Source Label",
                "Component",
                "Area",
                "Parent",
                "Children",
                "Max IoU",
                "Max Containment",
                "BBox",
            ]
        )
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(24)
        self.verticalHeader().setMinimumSectionSize(22)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setSortingEnabled(True)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._open_context_menu)
        self.itemDoubleClicked.connect(self._locate_item)
        self.setStyleSheet(
            """
            QTableWidget#candidateIsolationTable {
                background: #1f242c;
                alternate-background-color: #2a3038;
                color: #eef2f7;
                gridline-color: #3b444f;
                selection-background-color: #2f6f8f;
                selection-color: #ffffff;
            }
            QTableWidget#candidateIsolationTable::item {
                padding: 3px 4px;
            }
            QTableWidget#candidateIsolationTable::item:selected {
                background: #2f6f8f;
                color: #ffffff;
            }
            QHeaderView::section {
                background: #333941;
                color: #f3f6f9;
                border: 0;
                border-right: 1px solid #464e59;
                border-bottom: 1px solid #464e59;
                padding: 3px 5px;
                font-weight: 700;
            }
            """
        )

    def set_records(self, records: list[CandidateObjectRecord]) -> None:
        selected = set(self.selected_candidate_ids())
        self.setSortingEnabled(False)
        self.setRowCount(0)
        for record in records:
            row = self.rowCount()
            self.insertRow(row)
            self._set_numeric_item(row, 0, record.candidate_id)
            status_item = QTableWidgetItem(record.status)
            status_item.setToolTip(STATUS_TOOLTIPS.get(record.status, "Candidate status."))
            self.setItem(row, 1, status_item)
            self.setItem(row, 2, QTableWidgetItem(record.source_layer_name))
            self._set_numeric_item(row, 3, record.source_label_value)
            self._set_numeric_item(row, 4, record.source_component_id)
            self._set_numeric_item(row, 5, record.area)
            self._set_optional_numeric_item(row, 6, record.parent_candidate_id)
            self._set_numeric_item(row, 7, record.child_count)
            self._set_float_item(row, 8, record.max_iou)
            self._set_float_item(row, 9, record.max_containment)
            self.setItem(row, 10, QTableWidgetItem(record.bbox_text))
        self.setSortingEnabled(True)
        for candidate_id in selected:
            self.select_candidate_id(candidate_id, clear=False)

    def selected_candidate_ids(self) -> list[int]:
        ids: list[int] = []
        for index in self.selectionModel().selectedRows():
            candidate_id = self._candidate_id_for_row(index.row())
            if candidate_id is not None:
                ids.append(candidate_id)
        return ids

    def select_candidate_id(self, candidate_id: int, *, clear: bool = True) -> bool:
        if clear:
            self.clearSelection()
        for row in range(self.rowCount()):
            if self._candidate_id_for_row(row) == int(candidate_id):
                self.selectRow(row)
                self.scrollToItem(self.item(row, 0))
                return True
        return False

    def set_actions_enabled(self, enabled: bool) -> None:
        self._actions_enabled = bool(enabled)

    def _candidate_id_for_row(self, row: int) -> int | None:
        item = self.item(row, 0)
        if item is None:
            return None
        return int(item.data(Qt.UserRole))

    def _set_numeric_item(self, row: int, column: int, value: int) -> None:
        item = NumericTableWidgetItem(str(int(value)))
        item.setData(Qt.UserRole, int(value))
        self.setItem(row, column, item)

    def _set_optional_numeric_item(self, row: int, column: int, value: int | None) -> None:
        if value is None:
            item = QTableWidgetItem("")
        else:
            item = NumericTableWidgetItem(str(int(value)))
            item.setData(Qt.UserRole, int(value))
        self.setItem(row, column, item)

    def _set_float_item(self, row: int, column: int, value: float) -> None:
        item = NumericTableWidgetItem(f"{float(value):.3f}")
        item.setData(Qt.UserRole, float(value))
        self.setItem(row, column, item)

    def _open_context_menu(self, position) -> None:
        if not self._actions_enabled or self._action_callback is None or not self.selected_candidate_ids():
            return
        menu = QMenu(self)
        actions = {
            menu.addAction("Keep"): "keep",
            menu.addAction("Reject"): "reject",
            menu.addAction("Mark Duplicate"): "duplicate",
            menu.addAction("Mark Parent"): "parent",
            menu.addAction("Mark Child"): "nested_child",
            menu.addAction("Mark Fragment"): "fragment",
        }
        for action, status in actions.items():
            if status in STATUS_TOOLTIPS:
                action.setToolTip(STATUS_TOOLTIPS[status])
                action.setStatusTip(STATUS_TOOLTIPS[status])
        menu.addSeparator()
        actions[menu.addAction("Locate Candidate")] = "locate"
        actions[menu.addAction("Isolate Selected")] = "isolate_selected"
        selected = menu.exec_(self.viewport().mapToGlobal(position))
        if selected in actions:
            self._action_callback(actions[selected])

    def _locate_item(self, item: QTableWidgetItem) -> None:
        if self._locate_callback is None:
            return
        candidate_id = self._candidate_id_for_row(item.row())
        if candidate_id is not None:
            self._locate_callback(candidate_id)
