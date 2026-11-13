from types import SimpleNamespace

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication,QWidget

from napari_sam3_assistant.core.coordinates import infer_image_selection
from napari_sam3_assistant.core.models import ExemplarPrompt,PromptBundle,Sam3Result,Sam3Task
from napari_sam3_assistant.widgets.advanced.advanced_mode_panel import AdvancedModePanel
import napari_sam3_assistant.widgets.advanced.advanced_mode_panel as module


def test_full_image_scan_encodes_once_and_never_pastes_reference_pixels():
    app=QApplication.instance() or QApplication([])
    panel=AdvancedModePanel.__new__(AdvancedModePanel);QWidget.__init__(panel)
    data=np.arange(16*24,dtype=np.uint16).reshape(16,24)
    layer=SimpleNamespace(data=data,name='image')
    # The target tile can be smaller than the reference box: pixels are not pasted.
    ex=ExemplarPrompt(roi=data[2:12,2:12],y0=2.,x0=2.,y1=12.,x1=12.)
    bundle=PromptBundle(task=Sam3Task.EXEMPLAR,image=infer_image_selection('image',data.shape),exemplars=[ex])
    panel.viewer=SimpleNamespace(layers={'image':layer})
    panel.layer_writer=object()
    panel.activity_status=SimpleNamespace(set_starting_task=lambda:None,set_ready=lambda:None)
    panel._current_task=lambda:Sam3Task.EXEMPLAR
    panel._large_image_mode_enabled=lambda:True
    panel._folder_batch_enabled=lambda:False
    panel._z_stack_tiled_scan_enabled=lambda:False
    panel._selected_roi_size=lambda:(8,8)
    panel.tile_overlap_spin=SimpleNamespace(value=lambda:0)
    panel.merge_tile_seams_check=SimpleNamespace(isChecked=lambda:False)
    panel._tiled_exemplar_target_layer_names=lambda:['image']
    panel._external_exemplar_source_enabled=lambda:False
    panel._collect_bundle_for_tiled_exemplar=lambda _:bundle
    panel._cpu_bundle_support_error=lambda _:''
    panel._clear_results_table=lambda:None
    panel._show_active_roi_overlay=lambda *a,**kw:None
    calls={'encode':0,'target':0}
    class Adapter:
        def encode_exemplar(self,image,prompt):
            calls['encode']+=1
            np.testing.assert_array_equal(image,data)
            return SimpleNamespace(reference_hw=image.shape)
        def run_image_with_exemplar(self,tile,prompt,reference,**kwargs):
            calls['target']+=1
            assert tile.shape==(8,8)
            labels=np.zeros((8,8),dtype=np.uint32);labels[1:3,1:3]=1
            return Sam3Result(task=Sam3Task.EXEMPLAR,labels=labels,boxes_xyxy=np.asarray([[1,1,3,3]],dtype=np.float32),scores=np.asarray([.9]))
        def run_image(self,*args,**kwargs):raise AssertionError('Legacy pasted path was used')
    adapter=Adapter()
    panel._ensure_adapter=lambda:adapter
    panel._ensure_image_adapter_loaded_for_bundle=lambda *_:None
    output=[]
    panel._write_image_result=output.append
    panel._start_worker=lambda worker,**kwargs:worker.run()
    panel._log=lambda *_:None
    panel._run_batch_local_exemplar_task()
    assert calls=={'encode':1,'target':6}
    assert len(output)==1
    assert output[0].metadata['tile_prompting']=='encoded_native_reference'
    assert output[0].metadata['reference_hw']==(16,24)
    assert output[0].labels.shape==(16,24)
    assert len(np.unique(output[0].labels))==7
    panel.close()


def test_reference_box_straddling_grid_keeps_complete_context(monkeypatch):
    panel=AdvancedModePanel.__new__(AdvancedModePanel)
    source=SimpleNamespace(data=None)
    bundle=PromptBundle(task=Sam3Task.EXEMPLAR,image=infer_image_selection('image',(20000,20000)))
    ex=ExemplarPrompt(roi=None,y0=5000.25,x0=5000.75,y1=11500.5,x1=11500.25)
    seen=[]
    def extract(data,selection,bounds):
        seen.append(bounds)
        return np.broadcast_to(np.zeros((1,1),dtype=np.uint8),(bounds.height,bounds.width))
    monkeypatch.setattr(module,'extract_2d_roi',extract)
    class Adapter:
        def encode_exemplar(self,image,prompt):
            box=prompt.boxes[0]
            assert 0<=box.y0<box.y1<=image.shape[0]
            assert 0<=box.x0<box.x1<=image.shape[1]
            assert box.y1-box.y0==ex.y1-ex.y0
            assert box.x1-box.x0==ex.x1-ex.x0
            return 'reference'
    reference,bounds=panel._encode_scan_reference(Adapter(),source,bundle,None,ex)
    assert reference=='reference'
    assert panel._bounds_contain_exemplar(seen[0],ex)


class RecordingEncodedAdapter:
    def __init__(self, source):
        self.source = source
        self.reference = SimpleNamespace(reference_hw=source.shape[:2])
        self.encoded = []
        self.tiles = []

    def encode_exemplar(self, image, prompt):
        np.testing.assert_array_equal(image, self.source)
        box = prompt.boxes[0]
        assert box.y1 - box.y0 == 10  # Larger than the selected 8px target tile.
        self.encoded.append(np.asarray(image).copy())
        return self.reference

    def run_image_with_exemplar(self, tile, prompt, reference, *, cache_context):
        assert reference is self.reference
        assert tile.shape[:2] == (8, 8)
        self.tiles.append((tile.copy(), cache_context))
        labels = np.zeros((8, 8), dtype=np.uint32)
        labels[1:3, 1:3] = 1
        return Sam3Result(task=Sam3Task.EXEMPLAR, labels=labels)

    def run_image(self, *args, **kwargs):
        raise AssertionError('Full scans must not use legacy inference')


def scan_panel(source, selection):
    app = QApplication.instance() or QApplication([])
    panel = AdvancedModePanel.__new__(AdvancedModePanel)
    QWidget.__init__(panel)
    panel._test_app = app
    layer = SimpleNamespace(data=source, name='source', scale=(1,) * source.ndim)
    exemplar = ExemplarPrompt(roi=None, y0=2., x0=2., y1=12., x1=12.)
    bundle = PromptBundle(task=Sam3Task.EXEMPLAR, image=selection, exemplars=[exemplar])
    panel.viewer = SimpleNamespace(layers={'source': layer})
    panel.activity_status = SimpleNamespace(set_ready=lambda: None)
    panel._collect_bundle_for_tiled_exemplar = lambda: bundle
    panel._external_exemplar_source_enabled = lambda: False
    frame = source if source.ndim == 2 else source[selection.frame_index]
    panel._collect_tiled_exemplar_patch = lambda _: (frame[2:12, 2:12], 'source')
    panel._cpu_bundle_support_error = lambda _: ''
    panel._selected_roi_size = lambda: (8, 8)
    panel.tile_overlap_spin = SimpleNamespace(value=lambda: 0)
    panel.merge_tile_seams_check = SimpleNamespace(isChecked=lambda: False)
    panel._clear_results_table = lambda: None
    panel._ensure_image_adapter_loaded_for_bundle = lambda *_: None
    panel._start_worker = lambda worker, **kwargs: worker.run()
    panel._log = lambda *_: None
    return panel


def test_folder_scan_encodes_once_for_multiple_files_and_stack_slices(tmp_path):
    source = np.arange(16 * 24, dtype=np.uint16).reshape(16, 24)
    panel = scan_panel(source, infer_image_selection('source', source.shape))
    adapter = RecordingEncodedAdapter(source)
    panel._ensure_adapter = lambda: adapter
    input_dir = tmp_path / 'input'
    input_dir.mkdir()
    panel.folder_batch_input_edit = SimpleNamespace(text=lambda: str(input_dir))
    panel.folder_batch_output_edit = SimpleNamespace(text=lambda: str(tmp_path / 'out'))
    panel.folder_batch_format_combo = SimpleNamespace(currentText=lambda: 'TIFF')
    files = [input_dir / 'single.tif', input_dir / 'stack.tif', input_dir / 'rgb.png']
    panel._folder_batch_candidate_files = lambda _: files
    images = [source + 1000, np.stack([source + 2000, source + 3000]),
              np.stack([source + 4000, source + 5000, source + 6000], axis=-1)]
    panel._read_folder_batch_image = lambda path: images[files.index(path)]
    exports = []
    def export(labels, path, format):
        exports.append(labels.copy())
        return path
    panel.mask_export_service = SimpleNamespace(export=export)
    panel._run_folder_batch_exemplar_task()
    assert len(adapter.encoded) == 1
    assert len(adapter.tiles) == 24  # Four planes, six untouched tiles per plane.
    expected = []
    for plane in [images[0], *images[1], images[2]]:
        expected.extend(plane[y:y+8, x:x+8] for y in (0, 8) for x in (0, 8, 16))
    for (tile, _), original in zip(adapter.tiles, expected, strict=True):
        np.testing.assert_array_equal(tile, original)
    assert [mask.shape for mask in exports] == [(16, 24), (2, 16, 24), (16, 24)]
    assert all(np.count_nonzero(mask) == (48 if mask.ndim == 3 else 24) for mask in exports)
    panel.close()


def test_z_stack_scan_reuses_selected_slice_reference_and_writes_each_slice(tmp_path):
    from napari_sam3_assistant.huge_volume.mask_store import HugeVolumeMaskStore

    plane = np.arange(16 * 24, dtype=np.uint16).reshape(16, 24)
    source = np.stack([plane, plane + 1000])
    selection = infer_image_selection('source', source.shape, dims_current_step=(1, 0, 0))
    panel = scan_panel(source, selection)
    adapter = RecordingEncodedAdapter(source[1])
    panel._ensure_adapter = lambda: adapter
    panel.batch_all_images_check = SimpleNamespace(isChecked=lambda: False)
    path = tmp_path / 'z_stack.ome.zarr'
    panel._z_stack_mask_output_path = lambda _: path
    panel._run_z_stack_tiled_exemplar_task()
    assert len(adapter.encoded) == 1
    assert len(adapter.tiles) == 12
    for index, (tile, context) in enumerate(adapter.tiles):
        z_index, tile_index = divmod(index, 6)
        y_index, x_index = divmod(tile_index, 3)
        np.testing.assert_array_equal(tile, source[z_index, y_index*8:y_index*8+8, x_index*8:x_index*8+8])
        assert context['frame_index'] == z_index
    labels = np.asarray(HugeVolumeMaskStore.open(path).array)
    assert labels.shape == source.shape
    assert all(np.count_nonzero(frame) == 24 for frame in labels)
    assert panel._last_quick_mask_path == path
    panel.close()


def test_production_tile_requires_reference_instead_of_falling_back_to_legacy():
    panel = AdvancedModePanel.__new__(AdvancedModePanel)
    with pytest.raises(RuntimeError, match='Encode the source exemplar'):
        panel._infer_encoded_exemplar_tile(None, None, None, None, encoded_reference=None)
