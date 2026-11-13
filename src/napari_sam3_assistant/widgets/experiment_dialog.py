from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from threading import Event
from uuid import uuid4

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtCore import QUrl
from qtpy.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
    QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout)

from ..experiments import ExperimentPlan


class BatchExperimentDialog(QDialog):
    progress = Signal(str)
    load_requested = Signal(object, str)

    def __init__(self, parent, *, constants, minimum_size, tile_counts, start):
        super().__init__(parent)
        self.setObjectName('batchExperimentDialog')
        self.setStyleSheet('QDialog#batchExperimentDialog { background: #141821; color: #d8dee9; }')
        self.setWindowTitle('SAM3 Batch Experiment')
        self.resize(980, 650)
        self.constants = constants
        self.result_constants = constants
        self.start_callback = start
        self.cancel_event = Event()
        self.running = False
        self.folder = None
        layout = QVBoxLayout(self)
        explanation = QLabel('Variable: ROI size. The target image, exemplar, model, detection threshold, '
                             'overlap and seam-merge setting are captured once for the batch.')
        explanation.setWordWrap(True);layout.addWidget(explanation)
        self.fixed = QLabel(f'Target: {constants["image_layer"]}  |  Threshold: {constants["threshold"]:.2f}  |  '
                       f'Overlap: {constants["overlap_percent"]}%  |  Seam merge: {"on" if constants["merge_seams"] else "off"}')
        self.fixed.setWordWrap(True);layout.addWidget(self.fixed)
        sizes = QHBoxLayout();sizes.addWidget(QLabel('ROI sizes:'))
        self.size_checks = {}
        for size in (512,1024,2048,4096,8192):
            check = QCheckBox(str(size))
            check.setEnabled(size >= minimum_size)
            check.setChecked(size in (2048,4096,8192) and size >= minimum_size)
            check.setToolTip(f'{tile_counts[size]} tiles per series' if tile_counts[size] is not None else 'Tile count is calculated when a new batch source is selected')
            sizes.addWidget(check);self.size_checks[size]=check
        layout.addLayout(sizes)
        self.mode = QComboBox()
        self.mode.addItem('Encoded native reference — standard tile scan',('encoded',))
        self.mode.addItem('Compare encoded reference and legacy current',('encoded','current'))
        self.mode.addItem('Both legacy modes — separate comparison series',('current','reference'))
        self.mode.addItem('Legacy current: natural / pasted',('current',))
        self.mode.addItem('Legacy pasted-reference only',('reference',))
        layout.addWidget(self.mode)
        note = QLabel('Encoded scanning encodes the exemplar once in its native 8k scene and reuses those features on untouched tiles. '
                      'Legacy modes retain the old natural/pasted behavior for comparison. '
                      'Within each series only ROI size varies; the encoded source context remains fixed at 8k.')
        note.setWordWrap(True);layout.addWidget(note)
        self.save_tiles = QCheckBox('Also save decoded per-tile predictions (additional storage)')
        self.save_tiles.setToolTip('Target-tile labels, boxes and scores before full-image composition. Individual overlapping candidate masks are not retained.')
        layout.addWidget(self.save_tiles)
        folder_row=QHBoxLayout();folder_row.addWidget(QLabel('Save results in:'))
        self.output = QLineEdit(str(Path.home()/'SAM3 experiments'))
        self.browse = QPushButton('Choose')
        self.browse.clicked.connect(self.choose_folder)
        folder_row.addWidget(self.output);folder_row.addWidget(self.browse);layout.addLayout(folder_row)
        self.estimate = QLabel();self.estimate.setWordWrap(True);layout.addWidget(self.estimate)
        self.tile_counts=tile_counts
        for check in self.size_checks.values(): check.toggled.connect(self.update_estimate)
        self.mode.currentIndexChanged.connect(self.update_estimate)
        self.update_estimate()
        self.status=QLabel('Open experiment… to inspect saved results. A new batch requires an image and exemplar only when you click Start batch.')
        self.status.setWordWrap(True);layout.addWidget(self.status)
        self.progress.connect(self.status.setText)
        layout.addWidget(QLabel('Comparison before and after seam merging'))
        self.results=QTableWidget(0,10)
        self.results.setObjectName('resultsTable')
        self.results.setHorizontalHeaderLabels(['ROI','Prompting','Status','Seconds','Tiles','Empty tiles','Objects before','Objects after','Final coverage','Largest final mask (px)'])
        self.results.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results.setSelectionMode(QAbstractItemView.SingleSelection)
        self.results.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results.itemSelectionChanged.connect(self.sync_load)
        layout.addWidget(self.results)
        self.merge_summary=QLabel('Select a completed run to compare the two saved stages. Both masks are already stitched into full-image coordinates.')
        self.merge_summary.setWordWrap(True)
        layout.addWidget(self.merge_summary)
        result_row=QHBoxLayout()
        self.inspect_before=QPushButton('Inspect Before')
        self.inspect_before.setToolTip('Load the stitched full-image mask before seam merging. Keep the current viewer position and zoom.')
        self.inspect_after=QPushButton('Inspect After')
        self.inspect_after.setToolTip('Load the final full-image mask after seam merging. Keep the current viewer position and zoom.')
        self.inspect_before.setEnabled(False);self.inspect_after.setEnabled(False)
        self.inspect_before.clicked.connect(lambda: self.load_selected('composed_path'))
        self.inspect_after.clicked.connect(lambda: self.load_selected('final_path'))
        self.open_folder=QPushButton('Open results folder');self.open_folder.setEnabled(False)
        self.open_folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.folder))))
        self.open_existing = QPushButton('Open experiment…')
        self.open_existing.clicked.connect(self.open_experiment)
        result_row.addWidget(self.inspect_before);result_row.addWidget(self.inspect_after);result_row.addWidget(self.open_folder);result_row.addWidget(self.open_existing);layout.addLayout(result_row)
        buttons=QHBoxLayout()
        self.start_button=QPushButton('Start batch')
        self.start_button.clicked.connect(self.start_batch)
        self.cancel_button=QPushButton('Cancel batch');self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_batch)
        self.close_button=QPushButton('Close');self.close_button.clicked.connect(self.reject)
        for button in (self.start_button,self.cancel_button,self.close_button):buttons.addWidget(button)
        layout.addLayout(buttons)

    def choose_folder(self):
        folder=QFileDialog.getExistingDirectory(self,'Experiment output folder',self.output.text())
        if folder:self.output.setText(folder)

    def update_estimate(self,*_):
        sizes=[size for size,check in self.size_checks.items() if check.isChecked()]
        modes=self.mode.currentData()
        known=all(self.tile_counts[size] is not None for size in sizes)
        count=f'{sum(self.tile_counts[size] for size in sizes)*len(modes):,} tile inferences' if known else 'tile count available after source selection'
        self.estimate.setText(f'{len(sizes)*len(modes)} runs, {count}, processed sequentially. '
                              'Each run saves the composed mask, final mask when seam merging is enabled, and tile log. '
                              'Statistics describe instance separation and coverage, not biological accuracy.')

    def set_source_context(self,constants,tile_counts):
        self.constants=constants
        self.tile_counts=tile_counts
        self.fixed.setText(f'Target: {constants["image_layer"]}  |  Threshold: {constants["threshold"]:.2f}  |  '
                           f'Overlap: {constants["overlap_percent"]}%  |  Seam merge: {"on" if constants["merge_seams"] else "off"}')
        for size,check in self.size_checks.items():
            valid=size>=constants['minimum_size']
            if not valid:check.setChecked(False)
            check.setEnabled(valid and not self.running)
            check.setToolTip(f'{tile_counts[size]} tiles per series' if valid else 'Smaller than the complete exemplar crop')
        self.update_estimate()

    def start_batch(self):
        try:
            plan=ExperimentPlan(tuple(s for s,c in self.size_checks.items() if c.isChecked()),
                                tuple(self.mode.currentData()),self.constants['overlap_percent']/100.,
                                self.constants['merge_seams'],self.save_tiles.isChecked())
            if not self.output.text().strip():raise ValueError('Choose an output folder.')
            folder=Path(self.output.text()).expanduser()/f'experiment_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}'
            self.folder=folder;self.cancel_event=Event();self.running=True
            for widget in [self.start_button,self.output,self.browse,self.mode,self.save_tiles,self.open_existing,*self.size_checks.values()]:widget.setEnabled(False)
            self.cancel_button.setEnabled(True);self.results.setRowCount(0)
            self.sync_load()
            self.start_callback(plan,folder,self.cancel_event,self)
        except Exception as exc:
            self.finish({'status':'failed','error':str(exc),'runs':[]})

    def finish(self,manifest):
        self.result_constants = manifest.get('constants',self.constants)
        self.running=False;self.cancel_button.setEnabled(False)
        self.open_existing.setEnabled(True)
        self.start_button.setEnabled(True);self.output.setEnabled(True);self.browse.setEnabled(True);self.mode.setEnabled(True);self.save_tiles.setEnabled(True)
        # Restore only sizes that fit the exemplar.
        for size,check in self.size_checks.items():check.setEnabled(size >= self.constants['minimum_size'])
        self.status.setText(f'Batch {manifest["status"]}. '+str(manifest.get('error',''))+f' Results: {self.folder}')
        self.open_folder.setEnabled(self.folder is not None and self.folder.exists())
        self.results.setRowCount(len(manifest.get('runs',[])))
        for row,run in enumerate(manifest.get('runs',[])):
            stats=run.get('final_stats',{})
            mode_name={'encoded':'Encoded native','current':'Legacy current','reference':'Legacy reference'}.get(run['mode'],run['mode'])
            values=[str(run['roi_size']),mode_name,run['status'],f'{run.get("seconds",0):.1f}',
                    str(run.get('tile_count','—')),str(run.get('empty_tiles','—')),
                    str(run.get('composed_stats',{}).get('instances','—')),str(stats.get('instances','—')),
                    f'{stats["coverage"]:.1%}' if 'coverage' in stats else '—',str(stats.get('largest_instance_pixels','—'))]
            for col,value in enumerate(values):
                item=QTableWidgetItem(value);item.setData(Qt.UserRole,run)
                item.setToolTip(f'Natural tiles: {run.get("natural_tiles",0)}; pasted tiles: {run.get("reference_tiles",0)}; encoded tiles: {run.get("encoded_tiles",0)}; uniform tiles skipped: {run.get("skipped_uniform_tiles",0)}')
                self.results.setItem(row,col,item)
        self.results.resizeColumnsToContents()
        if self.results.rowCount():self.results.selectRow(0)
        self.sync_load()

    def open_experiment(self):
        path,_=QFileDialog.getOpenFileName(self,'Open experiment report',self.output.text(),'Experiment report (experiment.json)')
        if not path:return
        try:
            manifest=json.loads(Path(path).read_text(encoding='utf-8'))
            if manifest.get('format') != 'sam3-batch-experiment-v1':
                raise ValueError('This is not a SAM3 batch experiment report.')
            self.folder=Path(path).parent
            self.finish(manifest)
        except Exception as exc:
            self.status.setText(f'Cannot open experiment: {exc}')

    def sync_load(self):
        row=self.results.currentRow()
        item=self.results.item(row,0) if row>=0 else None
        run=item.data(Qt.UserRole) if item is not None else {}
        complete=not self.running and run.get('status')=='complete'
        self.inspect_before.setEnabled(complete and bool(run.get('composed_path')))
        self.inspect_after.setEnabled(complete and bool(run.get('final_path')))
        if not complete:
            self.merge_summary.setText('Select a completed run to compare the two saved stages. Both masks are already stitched into full-image coordinates.')
            return
        before=run.get('composed_stats',{})
        after=run.get('final_stats',{})
        if run.get('composed_path')==run.get('final_path'):
            message='Seam merging was not applied. Before and after show the same stitched mask.'
        elif 'instances' in before and 'instances' in after:
            n,m=before['instances'],after['instances']
            if n>m:message=f'Seam merging reduced {n:,} objects to {m:,}.'
            elif n==m:message=f'Seam merging kept the object count at {n:,}.'
            else:message=f'The object count changed from {n:,} to {m:,}.'
            if 'labeled_pixels' in before and before['labeled_pixels']==after.get('labeled_pixels'):
                message+=' Painted area did not change.'
        else:
            message='Object counts are unavailable for this saved experiment.'
        self.merge_summary.setText(message+' Before is already stitched; it excludes only seam merging.')

    def load_selected(self,stage):
        row=self.results.currentRow()
        item=self.results.item(row,0) if row>=0 else None
        if item is None or self.running:return
        run=item.data(Qt.UserRole)
        if run.get('status')=='complete' and run.get(stage):
            self.load_requested.emit(run,stage)

    def cancel_batch(self):
        self.cancel_event.set();self.status.setText('Cancelling after the current operation. Completed results will be kept.')
        self.cancel_button.setEnabled(False)

    def reject(self):
        if self.running:self.cancel_batch()
        else:super().reject()
