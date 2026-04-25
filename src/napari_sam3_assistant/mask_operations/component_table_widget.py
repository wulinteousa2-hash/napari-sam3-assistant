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

from .models import ComponentRecord


class NumericTableWidgetItem(QTableWidgetItem):
    def __lt__(self, other) -> bool:
        left = self.data(Qt.UserRole)
        right = other.data(Qt.UserRole)
        if left is not None and right is not None:
            return left < right
        return super().__lt__(other)


class ComponentTableWidget(QTableWidget):
    def __init__(
        self,
        delete_callback: Callable[[], None] | None = None,
        locate_callback: Callable[[int], None] | None = None,
    ) -> None:
        super().__init__(0, 6)
        self._delete_callback = delete_callback
        self._locate_callback = locate_callback
        self.setObjectName("componentAnalysisTable")
        self.setHorizontalHeaderLabels(["Component ID", "Label", "Area", "Centroid Y", "Centroid X", "BBox"])
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
            QTableWidget#componentAnalysisTable {
                background: #1f242c;
                alternate-background-color: #2a3038;
                color: #eef2f7;
                gridline-color: #3b444f;
                selection-background-color: #2f6f8f;
                selection-color: #ffffff;
            }
            QTableWidget#componentAnalysisTable::item {
                padding: 3px 4px;
            }
            QTableWidget#componentAnalysisTable::item:selected {
                background: #2f6f8f;
                color: #ffffff;
            }
            QTableWidget#componentAnalysisTable::item:alternate:selected {
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

    def set_records(self, records: list[ComponentRecord]) -> None:
        self.setSortingEnabled(False)
        self.setRowCount(0)
        for record in records:
            row = self.rowCount()
            self.insertRow(row)
            self._set_numeric_item(row, 0, record.component_id)
            self._set_numeric_item(row, 1, record.label_value)
            self._set_numeric_item(row, 2, record.area)
            self._set_float_item(row, 3, record.centroid_y)
            self._set_float_item(row, 4, record.centroid_x)
            self.setItem(row, 5, QTableWidgetItem(record.bbox_text))
        self.setSortingEnabled(True)

    def selected_component_ids(self) -> list[int]:
        ids: list[int] = []
        for index in self.selectionModel().selectedRows():
            component_id = self._component_id_for_row(index.row())
            if component_id is not None:
                ids.append(component_id)
        return ids

    def _component_id_for_row(self, row: int) -> int | None:
        item = self.item(row, 0)
        if item is None:
            return None
        return int(item.data(Qt.UserRole))

    def _set_numeric_item(self, row: int, column: int, value: int) -> None:
        item = NumericTableWidgetItem(str(value))
        item.setData(Qt.UserRole, int(value))
        self.setItem(row, column, item)

    def _set_float_item(self, row: int, column: int, value: float) -> None:
        item = NumericTableWidgetItem(f"{value:.2f}")
        item.setData(Qt.UserRole, float(value))
        self.setItem(row, column, item)

    def _open_context_menu(self, position) -> None:
        if self._delete_callback is None or not self.selected_component_ids():
            return
        menu = QMenu(self)
        action = menu.addAction("Delete Selected Components")
        if menu.exec_(self.viewport().mapToGlobal(position)) == action:
            self._delete_callback()

    def _locate_item(self, item: QTableWidgetItem) -> None:
        if self._locate_callback is None:
            return
        component_id = self._component_id_for_row(item.row())
        if component_id is not None:
            self._locate_callback(component_id)
