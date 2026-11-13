from qtpy.QtWidgets import QApplication

from napari_sam3_assistant.widgets.experiment_dialog import BatchExperimentDialog


def test_dialog_launches_separate_series_and_loads_completed_results(tmp_path):
    app=QApplication.instance() or QApplication([])
    launched=[]
    fixed={'image_layer':'image','threshold':.5,'overlap_percent':15,'merge_seams':True,'minimum_size':800}
    dialog=BatchExperimentDialog(None,constants=fixed,minimum_size=800,
        tile_counts={s:1 for s in (512,1024,2048,4096,8192)},start=lambda *args:launched.append(args))
    dialog.output.setText(str(tmp_path));dialog.start_batch()
    plan,folder,event,_=launched[0]
    assert plan.sizes==(2048,4096,8192)
    assert plan.modes==('encoded',)
    assert not dialog.mode.isEnabled()
    assert not dialog.open_existing.isEnabled()
    assert not dialog.size_checks[512].isEnabled()
    dialog.cancel_batch();assert event.is_set()
    dialog.finish({'status':'cancelled','constants':fixed,'runs':[{
        'name':'current_roi_2048','roi_size':2048,'mode':'current','status':'complete',
        'final_path':'current/final.ome.zarr','composed_path':'current/composed.ome.zarr',
        'composed_stats':{'instances':1290,'labeled_pixels':100},
        'final_stats':{'instances':12,'labeled_pixels':100,'coverage':.25,'largest_instance_pixels':20}}]})
    loaded=[];dialog.load_requested.connect(lambda *args:loaded.append(args))
    assert dialog.inspect_before.isEnabled() and dialog.inspect_after.isEnabled()
    assert dialog.results.item(0,6).text()=='1290'
    assert dialog.results.item(0,7).text()=='12'
    assert '1,290 objects to 12' in dialog.merge_summary.text()
    assert 'Painted area did not change' in dialog.merge_summary.text()
    dialog.inspect_before.click();dialog.inspect_after.click()
    assert loaded[0][0]['name']=='current_roi_2048'
    assert [entry[1] for entry in loaded]==['composed_path','final_path']
    assert not dialog.size_checks[512].isEnabled()
    dialog.close()


def test_inspection_handles_unmerged_and_incomplete_runs():
    app=QApplication.instance() or QApplication([])
    fixed={'image_layer':'image','threshold':.5,'overlap_percent':15,'merge_seams':False,'minimum_size':800}
    dialog=BatchExperimentDialog(None,constants=fixed,minimum_size=800,
        tile_counts={s:1 for s in (512,1024,2048,4096,8192)},start=lambda *_:None)
    dialog.finish({'status':'failed','runs':[
        {'name':'done','roi_size':2048,'mode':'current','status':'complete',
         'composed_path':'mask','final_path':'mask','composed_stats':{'instances':3},'final_stats':{'instances':3}},
        {'name':'failed','roi_size':4096,'mode':'current','status':'failed'}]})
    assert 'Seam merging was not applied' in dialog.merge_summary.text()
    dialog.results.selectRow(1)
    assert not dialog.inspect_before.isEnabled() and not dialog.inspect_after.isEnabled()
    loaded=[];dialog.load_requested.connect(lambda *args:loaded.append(args))
    dialog.load_selected('final_path')
    assert not loaded
    dialog.close()
