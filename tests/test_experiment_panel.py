import json
from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from napari.layers import Image
from qtpy.QtWidgets import QApplication, QWidget

from napari_sam3_assistant.adapters.sam3_backend import Sam3AdapterConfig
from napari_sam3_assistant.core.coordinates import infer_image_selection
from napari_sam3_assistant.core.models import ExemplarPrompt, PromptBundle, Sam3Result, Sam3Task
from napari_sam3_assistant.widgets.advanced.advanced_mode_panel import AdvancedModePanel
from napari_sam3_assistant.widgets.experiment_dialog import BatchExperimentDialog
from napari_sam3_assistant.experiments import create_mask,finish_mask
from threading import Event
from qtpy.QtWidgets import QFileDialog


@pytest.mark.parametrize('encoded',[False,True])
def test_panel_snapshots_pixels_and_runs_all_modes_without_ui_reads_in_worker(tmp_path,monkeypatch,encoded):
    app=QApplication.instance() or QApplication([])
    panel=AdvancedModePanel.__new__(AdvancedModePanel)
    QWidget.__init__(panel)
    data=np.arange(32*32,dtype=np.uint16).reshape(32,32)
    original=data.copy()
    layer=Image(data,name='source')
    exemplar=ExemplarPrompt(roi=data[8:12,9:14].copy(),y0=8.,x0=9.,y1=12.,x1=14.)
    bundle=PromptBundle(task=Sam3Task.EXEMPLAR,image=infer_image_selection('source',data.shape),exemplars=[exemplar])
    panel.viewer=SimpleNamespace(layers={'source':layer})
    panel._worker=None
    panel._current_image_layer_name=lambda:'source'
    panel._current_task=lambda:Sam3Task.EXEMPLAR
    panel._large_image_mode_enabled=lambda:True
    panel._collect_bundle_for_tiled_exemplar=lambda:bundle
    panel._external_exemplar_source_enabled=lambda:False
    panel._cpu_bundle_support_error=lambda _:''
    panel.tile_overlap_spin=SimpleNamespace(value=lambda:15)
    panel.merge_tile_seams_check=SimpleNamespace(isChecked=lambda:True)
    config=Sam3AdapterConfig(device='cpu')
    panel._adapter_config=lambda:config
    calls=[]
    reference_calls=[]
    class FakeAdapter:
        def __init__(self):self.config=config
        def encode_exemplar(self,canvas,prompt):
            reference_calls.append(np.asarray(canvas).copy())
            return SimpleNamespace(reference_hw=canvas.shape[:2])
        def run_image_with_exemplar(self,canvas,prompt,reference,**kwargs):
            return self.run_image(canvas,prompt,**kwargs)
        def run_image(self,canvas,prompt,*,cache_context=None):
            calls.append(np.asarray(canvas).copy())
            # Mutating the live source must not affect later experiment runs.
            data[:]=0
            labels=np.zeros(canvas.shape[:2],dtype=np.uint32)
            labels[1:3,-7:-5]=1
            return Sam3Result(task=Sam3Task.EXEMPLAR,labels=labels,object_ids=np.asarray([1]))
    adapter=FakeAdapter()
    panel._ensure_adapter=lambda:adapter
    panel._ensure_image_adapter_loaded_for_bundle=lambda *_:None
    panel._start_worker=lambda worker,**kwargs:worker.run()
    panel._log=lambda *_:None
    finished=[]
    def execute(dialog):
        dialog.mode.setCurrentIndex(0 if encoded else 2)
        dialog.output.setText(str(tmp_path))
        for size,check in dialog.size_checks.items():check.setChecked(size in (512,1024))
        dialog.start_batch()
        assert not dialog.running
        finished.append(dialog.folder)
        return 0
    monkeypatch.setattr(BatchExperimentDialog,'exec',execute)
    panel._open_batch_experiment()
    folder=finished[0]
    manifest=json.loads((folder/'experiment.json').read_text())
    assert manifest['status']=='complete'
    assert len(manifest['runs'])==(2 if encoded else 4)
    assert len(calls)==(2 if encoded else 4)
    assert len(reference_calls)==(1 if encoded else 0)
    snapshot=zarr.open_group(str(folder/'source.zarr'),mode='r')['image']
    np.testing.assert_array_equal(np.asarray(snapshot),original)
    for canvas in calls:
        np.testing.assert_array_equal(canvas[:32,-32:],original)
    assert [r['natural_tiles'] for r in manifest['runs']]==([0,0] if encoded else [1,1,0,0])
    assert [r['reference_tiles'] for r in manifest['runs']]==([0,0] if encoded else [0,0,1,1])
    if encoded:assert all(r['encoded_tiles']==1 for r in manifest['runs'])
    assert all(r['final_stats']['instances']==1 for r in manifest['runs'])
    panel._batch_experiment_dialog.close()
    panel.close()


def test_open_saved_experiment_without_source_exemplar_or_model(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    panel=AdvancedModePanel.__new__(AdvancedModePanel)
    QWidget.__init__(panel)
    class Layers(dict):
        def __iter__(self):return iter(self.values())
    layers=Layers()
    loaded=[]
    def add_labels(data,**kwargs):
        layer=SimpleNamespace(data=data,visible=True,**kwargs)
        layers[layer.name]=layer
        loaded.append(layer)
        return layer
    panel.viewer=SimpleNamespace(layers=layers,add_labels=add_labels,
        camera=SimpleNamespace(center=(7,8),zoom=3.5))
    panel._worker=None
    panel._current_image_layer_name=lambda:''
    panel._current_task=lambda:Sam3Task.EXEMPLAR
    panel._large_image_mode_enabled=lambda:True
    panel.tile_overlap_spin=SimpleNamespace(value=lambda:15)
    panel.merge_tile_seams_check=SimpleNamespace(isChecked=lambda:True)
    calls=[]
    def no_prompts():
        calls.append('prompts')
        raise RuntimeError('Draw at least one exemplar box')
    panel._collect_bundle_for_tiled_exemplar=no_prompts
    panel._adapter_config=lambda: (_ for _ in ()).throw(AssertionError('Model validation must not run'))
    panel._log=lambda *_:None
    run={'name':'current_roi_2048','roi_size':2048,'mode':'current','status':'complete',
         'composed_path':'before.ome.zarr','final_path':'after.ome.zarr'}
    for stage,value in [('composed',3),('final',1)]:
        group,array=create_mask(tmp_path/run[f'{stage}_path'],(4,4))
        array[:]=np.asarray([[1,1,0,0],[1,1,0,0],[0,0,value,value],[0,0,value,value]],dtype=np.uint32)
        run[f'{stage}_stats']=finish_mask(group,Event())
    manifest={'format':'sam3-batch-experiment-v1','status':'complete',
              'constants':{'source_transform':{'scale':[2.,3.],'translate':[4.,5.]}},'runs':[run]}
    report=tmp_path/'experiment.json';report.write_text(json.dumps(manifest))
    monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *args,**kwargs:(str(report),''))
    def execute(dialog):
        assert not calls
        dialog.open_experiment()
        assert dialog.inspect_before.isEnabled() and dialog.inspect_after.isEnabled()
        dialog.inspect_before.click();dialog.inspect_after.click()
        assert len(loaded)==2
        assert not loaded[0].visible and loaded[1].visible
        assert loaded[0].scale==[2.,3.] and loaded[1].translate==[4.,5.]
        assert np.asarray(loaded[0].data).max()==3
        assert np.asarray(loaded[1].data).max()==1
        assert panel.viewer.camera.center==(7,8) and panel.viewer.camera.zoom==3.5
        assert not calls
        # Starting a NEW batch still validates prompts, without blocking review.
        dialog.output.setText(str(tmp_path/'new'))
        dialog.start_batch()
        assert calls==['prompts']
        assert not dialog.running
        assert 'Draw at least one exemplar box' in dialog.status.text()
        assert dialog.open_existing.isEnabled()
        assert not (tmp_path/'new').exists()
        return 0
    monkeypatch.setattr(BatchExperimentDialog,'exec',execute)
    panel._open_batch_experiment()
    panel._batch_experiment_dialog.close();panel.close()
