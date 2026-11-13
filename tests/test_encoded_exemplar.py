from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from napari_sam3_assistant.adapters import Sam3Adapter,Sam3AdapterConfig
from napari_sam3_assistant.core.coordinates import infer_image_selection
from napari_sam3_assistant.core.models import BoxPrompt,PromptBundle,Sam3Result,Sam3Task


def bundle(data):
    return PromptBundle(task=Sam3Task.EXEMPLAR,image=infer_image_selection('image',data.shape),
                        boxes=[BoxPrompt(y0=1.,x0=1.,y1=3.,x1=3.)])


class Geometry(torch.nn.Module):
    def forward(self,geo_prompt=None,img_feats=None,img_sizes=None,img_pos_embeds=None):
        return torch.full((2,1,4),float(img_feats[0].mean())),torch.zeros((1,2),dtype=torch.bool)


def fake_adapter():
    adapter=Sam3Adapter(Sam3AdapterConfig(device='cpu'))
    class Model:
        def __init__(self):
            self.geometry_encoder=Geometry()
            self.backbone=SimpleNamespace(forward_text=lambda *a,**kw:{})
            self.seen=[]
            self.fail=False
        def _get_dummy_prompt(self):
            return SimpleNamespace(append_boxes=lambda *_:None)
        def _get_img_feats(self,state,ids):
            return state,[state['image']],[torch.zeros(1)],[state['image'].shape]
        def forward_grounding(self,backbone_out,**kwargs):
            value=self.geometry_encoder(img_feats=[backbone_out['image']])
            self.seen.append(value[0].clone())
            if self.fail:raise RuntimeError('forward failed')
            return {}
    adapter.image_model=Model()
    adapter.image_processor=SimpleNamespace(device='cpu',find_stage=SimpleNamespace(img_ids=torch.tensor([0])))
    adapter._inference_context=lambda:nullcontext()
    adapter._initial_state_for_image=lambda rgb,*a,**kw:{'backbone_out':{'image':torch.from_numpy(rgb[...,0]).float()}}
    adapter._normalize_state_tensors=lambda *_:None
    adapter._decode_encoded_tile=lambda outputs,hw,b:Sam3Result(task=b.task,labels=np.ones(hw,dtype=np.uint32))
    return adapter


def test_features_are_encoded_once_reused_and_hook_removed():
    adapter=fake_adapter()
    source=np.arange(16,dtype=np.uint8).reshape(4,4)
    reference=adapter.encode_exemplar(source,bundle(source))
    expected=reference.features.clone()
    source[:]=0
    for value in (10,30):
        tile=np.arange(value,value+16,dtype=np.uint8).reshape(4,4)
        result=adapter.run_image_with_exemplar(tile,bundle(tile),reference)
        assert result.metadata['encoded_exemplar']
        assert result.labels.shape==tile.shape
        torch.testing.assert_close(adapter.image_model.seen[-1],expected)
        assert not adapter.image_model.geometry_encoder._forward_hooks
    assert len(adapter.image_model.seen)==2


def test_exception_cannot_leak_reference_hook():
    adapter=fake_adapter();data=np.arange(16,dtype=np.uint8).reshape(4,4)
    reference=adapter.encode_exemplar(data,bundle(data))
    adapter.image_model.fail=True
    with pytest.raises(RuntimeError,match='forward failed'):
        adapter.run_image_with_exemplar(data,bundle(data),reference)
    assert not adapter.image_model.geometry_encoder._forward_hooks


def test_reference_invalid_after_config_change_or_unload():
    adapter=fake_adapter();data=np.arange(16,dtype=np.uint8).reshape(4,4)
    reference=adapter.encode_exemplar(data,bundle(data))
    adapter.config.confidence_threshold=.35
    with pytest.raises(RuntimeError,match='stale'):
        adapter.run_image_with_exemplar(data,bundle(data),reference)
    adapter.config.confidence_threshold=.5
    adapter.unload()
    with pytest.raises(RuntimeError,match='stale'):
        adapter.run_image_with_exemplar(data,bundle(data),reference)


def test_reference_cannot_be_used_by_another_adapter():
    adapter=fake_adapter();other=fake_adapter();data=np.arange(16,dtype=np.uint8).reshape(4,4)
    reference=adapter.encode_exemplar(data,bundle(data))
    with pytest.raises(RuntimeError,match='stale'):
        other.run_image_with_exemplar(data,bundle(data),reference)


def test_uniform_padding_does_not_hallucinate_components():
    adapter=fake_adapter();data=np.arange(16,dtype=np.uint8).reshape(4,4)
    reference=adapter.encode_exemplar(data,bundle(data))
    result=adapter.run_image_with_exemplar(np.zeros_like(data),bundle(data),reference)
    assert result.metadata['skipped_uniform_tile']
    assert not np.any(result.labels)
    assert not adapter.image_model.seen


@pytest.mark.parametrize('empty',[False,True])
def test_streamed_projection_matches_original_bulk_mask_projection(empty):
    from sam3.model import box_ops
    from sam3.model.data_misc import interpolate
    adapter=Sam3Adapter(Sam3AdapterConfig(device='cpu'))
    data=np.zeros((5,7),dtype=np.uint8)
    outputs={
        'pred_logits':torch.tensor([[[-10.],[-20.]]]) if empty else torch.tensor([[[2.],[2.]]]),
        'presence_logit_dec':torch.tensor([[5.]]),
        'pred_boxes':torch.tensor([[[.5,.5,.6,.6],[.25,.25,.5,.5]]]),
        'pred_masks':torch.tensor([[[[1.,1.],[-1.,-1.]],[[-1.,1.],[-1.,1.]]]]),
    }
    result=adapter._decode_encoded_tile(outputs,data.shape,bundle(data))
    scores=(outputs['pred_logits'].sigmoid()*outputs['presence_logit_dec'].sigmoid().unsqueeze(1)).squeeze(-1)
    keep=scores>.5
    masks=outputs['pred_masks'][keep]
    expected=np.zeros(data.shape,dtype=np.uint32)
    if len(masks):
        bulk=interpolate(masks.unsqueeze(1),data.shape,mode='bilinear',align_corners=False).sigmoid()>.5
        for index,mask in enumerate(bulk[:,0]):expected[mask.numpy()]=index+1
    np.testing.assert_array_equal(result.labels,expected)
    np.testing.assert_allclose(result.scores,scores[keep].numpy())
    expected_boxes=box_ops.box_cxcywh_to_xyxy(outputs['pred_boxes'][keep])*torch.tensor([7,5,7,5])
    np.testing.assert_allclose(result.boxes_xyxy,expected_boxes.numpy())
