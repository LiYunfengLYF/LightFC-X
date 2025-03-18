import os
import sys
import time

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import torch
from thop import profile, clever_format

from lib.models.lightfcx.trackerModel_lightfc import TRACKER_REGISTRY
from lib.utils.load import load_yaml, load_pretrain

z = torch.rand(1, 3, 128, 128).cuda()
zf = torch.rand(1, 320, 8, 8).cuda()
x = torch.rand(1, 3, 256, 256).cuda()

'''
RGBD_baseline_ep45     6.678M
RGBD_baseline_N3_ep45  6.826M
RGBE_baseline_ep45     6.678M
RGBT_baseline_ep45     6.678M
'''
cfg = load_yaml(rf'E:\code\LightFCX-review/experiments/lightfcx/RGBD_baseline_N3_ep45.yaml')
model = TRACKER_REGISTRY.get(cfg.MODEL.NETWORK)(cfg, env_num=101, training=True).cuda()
# model.load_state_dict(torch.load(r'E:\code\LightFCX-review\output\checkpoints\train\lightfcx\RGBS_baseline_ep30\RGBSer_lightfc_ep0030.pth.tar')['net'])
model_input = ([z, z], [x, x])
for module in model.backbone.modules():
    if hasattr(module, 'switch_to_deploy'):
        module.switch_to_deploy()
if hasattr(model.backbone, 'finetune_tracking'):
    model.backbone.finetune_tracking()

for module in model.fusion.modules():
    if hasattr(module, 'switch_to_deploy'):
        module.switch_to_deploy()
for module in model.head.modules():
    if hasattr(module, 'switch_to_deploy'):
        module.switch_to_deploy()

with torch.no_grad():
    macs, params = profile(model, inputs=model_input, custom_ops=None, verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('overall Model macs is ', macs)
    print('overall Model params is ', params)

'''
RGBD_baseline_N3_update_ep15     7.762M
RGBE_baseline_update_ep15        7.614M
RGBT_baseline_update_ep15        7.614M
'''
updete_cfg = load_yaml(rf'E:\code\LightFCX-review/experiments/lightfcx/RGBE_baseline_update_ep15.yaml')
update_model = TRACKER_REGISTRY.get(updete_cfg.MODEL.NETWORK)(updete_cfg, env_num=101, training=True).cuda()
# model.load_state_dict(torch.load(r'E:\code\LightFCX-review\output\checkpoints\train\lightfcx\RGBS_baseline_ep30\RGBSer_lightfc_ep0030.pth.tar')['net'])
update_model_input = ([[z, z], [z, z],], [x, x])
for module in update_model.backbone.modules():
    if hasattr(module, 'switch_to_deploy'):
        module.switch_to_deploy()
if hasattr(update_model.backbone, 'finetune_tracking'):
    update_model.backbone.finetune_tracking()

for module in update_model.fusion.modules():
    if hasattr(module, 'switch_to_deploy'):
        module.switch_to_deploy()
for module in update_model.head.modules():
    if hasattr(module, 'switch_to_deploy'):
        module.switch_to_deploy()

with torch.no_grad():
    macs, params = profile(update_model, inputs=update_model_input, custom_ops=None, verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('overall update_model macs is ', macs)
    print('overall update_model params is ', params)


'''
RGBS_baseline_ep30     10.396M
'''
rgbs_cfg = load_yaml(rf'E:\code\LightFCX-review/experiments/lightfcx/RGBS_baseline_ep30.yaml')
rgbs_model = TRACKER_REGISTRY.get(rgbs_cfg.MODEL.NETWORK)(rgbs_cfg, env_num=101, training=True).cuda()
# model.load_state_dict(torch.load(r'E:\code\LightFCX-review\output\checkpoints\train\lightfcx\RGBS_baseline_ep30\RGBSer_lightfc_ep0030.pth.tar')['net'])
rgbs_model_input = ([z, z], [x, x])
for module in rgbs_model.backbone.modules():
    if hasattr(module, 'switch_to_deploy'):
        module.switch_to_deploy()
if hasattr(rgbs_model.backbone, 'finetune_tracking'):
    rgbs_model.backbone.finetune_tracking()

for module in rgbs_model.fusion.modules():
    if hasattr(module, 'switch_to_deploy'):
        module.switch_to_deploy()
for module in rgbs_model.head.modules():
    if hasattr(module, 'switch_to_deploy'):
        module.switch_to_deploy()

with torch.no_grad():
    macs, params = profile(rgbs_model, inputs=rgbs_model_input, custom_ops=None, verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('overall RGBS_model macs is ', macs)
    print('overall RGBS_model params is ', params)
