import json
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from napari_sam3_assistant.experiments import (
    ExperimentPlan, create_mask, finish_mask, merge_mask, run_experiment,
)
from napari_sam3_assistant.core.coordinates import RoiBounds
from napari_sam3_assistant.widgets.advanced.advanced_mode_panel import AdvancedModePanel


def callbacks():
    panel=AdvancedModePanel.__new__(AdvancedModePanel)
    def compose(existing,labels,next_id):
        return panel._compose_tile_labels(existing,labels,
            RoiBounds(y0=0,x0=0,y1=labels.shape[0],x1=labels.shape[1]),next_id)
    return dict(tile_plan=panel._tile_bounds_for_image,compose=compose,
                vertical_pairs=panel._vertical_seam_label_pairs,
                horizontal_pairs=panel._horizontal_seam_label_pairs)


def test_experiment_runs_both_series_and_preserves_each_stage(tmp_path):
    seen=[]
    def infer(b,mode):
        seen.append((b,mode))
        labels=np.ones((b.height,b.width),dtype=np.uint32)
        return labels,np.zeros((0,4)),None,'natural' if mode=='current' else 'reference',{'canvas_shape':list(labels.shape)}
    result=run_experiment(folder=tmp_path,plan=ExperimentPlan((6,8),('current','reference'),.25,True,True),
        shape=(12,12),constants={'threshold':.5,'exemplar_box':[1,1,3,3]},infer=infer,cancel=Event(),**callbacks())
    assert result['status']=='complete'
    assert [(r['mode'],r['roi_size']) for r in result['runs']]==[('current',6),('current',8),('reference',6),('reference',8)]
    for run in result['runs']:
        assert run['status']=='complete'
        assert run['final_stats']['coverage']==1
        assert run['composed_stats']['instances'] > run['final_stats']['instances']
        assert (tmp_path/run['composed_path']).exists()
        assert (tmp_path/run['final_path']).exists()
        entries=[json.loads(s) for s in (tmp_path/run['name']/'tiles.jsonl').read_text().splitlines()]
        assert len(entries)==run['tile_count']
        assert all(e['prompt_path']==('natural' if run['mode']=='current' else 'reference') for e in entries)
        assert len(list((tmp_path/run['name']/'tiles').glob('*.npz')))==run['tile_count']
    assert len(seen)==sum(r['tile_count'] for r in result['runs'])
    assert len((tmp_path/'comparison.csv').read_text().splitlines())==5
    assert json.loads((tmp_path/'experiment.json').read_text())['constants']['threshold']==.5


@pytest.mark.parametrize('touching',[True,False])
def test_chunked_merge_matches_normal_scanning_exactly(tmp_path,touching):
    panel=AdvancedModePanel.__new__(AdvancedModePanel)
    data=np.zeros((12,12),dtype=np.uint32)
    data[2:10,2:6]=7;data[2:10,6:10]=12;data[0,0]=2
    if not touching:
        data[:]=0;data[2:4,2:4]=7;data[8:10,8:10]=12;data[0,0]=2
    tiles=panel._tile_bounds_for_image(data.shape,(6,6),0)
    expected,count=panel._merge_tile_seam_labels(data,tiles,dilation_px=2,min_contact_pixels=8)
    _,source=create_mask(tmp_path/'source',data.shape);source[:]=data
    _,dest=create_mask(tmp_path/'dest',data.shape)
    actual_count=merge_mask(source,dest,tiles,panel._vertical_seam_label_pairs,panel._horizontal_seam_label_pairs,Event())
    np.testing.assert_array_equal(np.asarray(dest),expected)
    assert actual_count==count


def test_cancel_keeps_completed_runs(tmp_path):
    event=Event()
    def infer(b,mode):return np.zeros((b.height,b.width),dtype=np.uint32),np.zeros((0,4)),None,'reference',{}
    def progress(message):
        if message.startswith('Completed'):event.set()
    result=run_experiment(folder=tmp_path,plan=ExperimentPlan((6,8),('reference',),0,False),shape=(12,12),
        constants={},infer=infer,cancel=event,progress=progress,**callbacks())
    assert result['status']=='cancelled'
    assert len(result['runs'])==1 and result['runs'][0]['status']=='complete'
    assert (tmp_path/result['runs'][0]['final_path']).exists()


def test_failed_inference_is_not_reported_as_an_empty_tile(tmp_path):
    def infer(*_):raise RuntimeError('model failed')
    result=run_experiment(folder=tmp_path,plan=ExperimentPlan((6,),('current',),0,False),shape=(12,12),
        constants={},infer=infer,cancel=Event(),**callbacks())
    assert result['status']=='failed'
    assert result['runs'][0]['status']=='failed'
    assert result['runs'][0]['empty_tiles']==0
    assert result['runs'][0]['tiles_completed']==0
    assert 'model failed' in result['error']


def test_label_pyramid_retains_nonzero_edge_pixels(tmp_path):
    group,array=create_mask(tmp_path/'mask',(1025,3))
    array[0,1]=7;array[1,0]=8;array[1024,2]=9
    stats=finish_mask(group,Event())
    assert stats['instances']==3 and stats['labeled_pixels']==3
    assert group['s1'][0,0]==7
    assert group['s1'][-1,-1]==9
    assert len(group.attrs['multiscales'][0]['datasets'])==2


@pytest.mark.parametrize('sizes,modes',[((),('current',)),((4,4),('current',)),((4,),()),((4,),('other',))])
def test_plan_rejects_invalid_series(sizes,modes):
    with pytest.raises(ValueError):ExperimentPlan(sizes,modes,0,False)
