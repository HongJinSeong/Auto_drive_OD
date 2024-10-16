from mmdet.apis import init_detector
import mmcv
from mmcv import Config


import copy
import os.path as osp

import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

from mmdet.apis import set_random_seed


import json

import glob as _glob
import os
    
def glob(dir, pats, recursive=False):  # faster than match, python3 only
    pats = pats if isinstance(pats, (list, tuple)) else [pats]
    matches = []
    for pat in pats:
        matches += _glob.glob(os.path.join(dir, pat), recursive=recursive)
    return matches

#### load_annotations에서 뒤의 변수 받는거 custom dataset 에서는 이름을 바꿔도 되지만 아래에
#### configuration에서는 무조건 변수명을 ann_file로 받아야함
@DATASETS.register_module()
class Drive_dataset(CustomDataset):
    CLASSES=('car','bus','truck', 'special vehicle', 'motorcycle','bicycle','personal mobility','person','Traffic_light', 'Traffic_sign')


    def load_annotations(self, ann_fol):
        
        CLASSES_dict = {'car' : 0 , 'bus' : 1, 'truck' : 2, 'special vehicle' : 3, 'motorcycle' : 4,'bicycle' : 5 ,'personal mobility' : 6 
                        ,'person' : 7 ,'Traffic_light' : 8, 'Traffic_sign' : 9}
        
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        
        data_infos = []
        
        ls = glob(ann_fol,'*',True)
        
        for idx,an in enumerate(ls):
            json_data = {}
            with open(an, "r") as json_file:
                json_data = json.load(json_file)
                
            ansplit = an.split('/')
            
            filename = ansplit[0] + '/' + ansplit[1] + '/' + 'images'+'/'+ json_data['image_name']
            
            width, height = json_data['image_size']

            data_info = dict(filename=filename, width=width, height=height)

            gt_bboxes = []
            gt_labels = []

            for ann_data in json_data['Annotation']:
                gt_labels.append(CLASSES_dict[ann_data['class_name']])
                gt_bboxes.append(ann_data['data'])


            data_anno = dict(
                    bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                    labels=np.array(gt_labels, dtype=np.long))


            data_info.update(ann=data_anno)
            data_infos.append(data_info)
            
            if idx!=0 and idx%20000==0:
                print(str(idx)+'/'+str(len(ls))+' load annotations END!')
            
        
        
        return data_infos

## 추가수정 기존 받았던 pretrain과 매칭되는 config로 수정 
cfg = Config.fromfile('UniverseNet/configs/waymo_open/universenet50_2008_fp16_4x4_mstrain_640_1280_1x_waymo_open_f0.py') 

## 추가 및 수정 ## 
cfg.dataset_type  = 'Drive_dataset'
cfg.data_root = ''

## single GPU 이기 때문에 syncBN 이 아닌 BN으로 수정)
cfg.model.backbone.norm_cfg=dict(type='BN', requires_grad=True)

## Validation pipeline에 train pipeline 적용하기 위해서 구성 
cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=(1920, 1200),
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    
]

### test pipeline 나중에 test진행에 사용할 거 실제 validation은 위의 pipeline 으로 진행
cfg.test_pipeline = [
    ### TSET때 사용할 test time augmentation용 pipeline
    dict(type='LoadImageFromFile'),
    dict(
                type='MultiScaleFlipAug',
                img_scale=(1920, 1200),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                      dict(type='Collect', keys=['img'])
                ])
]


cfg.data=dict(
    samples_per_gpu=10, # batchsize
    workers_per_gpu=12, # batch를 불러오기 위한 작업 thread 갯수 
    # train dataset 
    train=dict(
        type=cfg.dataset_type,
        ann_file='2DBB/training/labels/',
        pipeline=cfg.train_pipeline),
    # validation dataset  
    val=dict(
        type=cfg.dataset_type,
        ann_file='2DBB/validation/labels',
        pipeline=cfg.test_pipeline),
    test=dict(
        type=cfg.dataset_type,
        ann_file='2DBB/test/labels/',
        pipeline=cfg.test_pipeline),)

## class 갯수 
cfg.model.bbox_head.num_classes=10

## GPU 학습 진행을 위한 device 선언
cfg.device='cuda'

## weight 와 학습 log 저장 위치 
cfg.work_dir = 'checkpoints_Best_ver1'

## log interval
cfg.log_config.interval = 8000 #iteration 단위

cfg.seed = 2024

## seed 고정 진행
set_random_seed(cfg.seed, deterministic=False)

cfg.workflow = [('train', 1), ('val',1)]

cfg.evaluation = dict(interval=1, metric='mAP')

### coco dataset으로 pretrain된 weight load 할 path
cfg.load_from = 'universenet50_2008_fp16_4x4_mstrain_480_960_2x_coco_20200815_epoch_24-81356447.pth'

### epoch 선언
cfg.runner = dict(type='EpochBasedRunner', max_epochs=50)

### 사용할 GPU 선언
cfg.gpu_ids = range(1)


# Build the detector
#model = init_detector(config, checkpoint, device='cuda:0')
checkpoint='checkpoints_Best_ver1_more/epoch_80.pth'
# model = init_detector(cfg,checkpoint, device='cpu')
model = init_detector(cfg, checkpoint, device='cuda:0')
model = model.eval()

from mmdet.apis import init_detector, inference_detector
import cv2
import os

import csv
csv.register_dialect(
    'mydialect',
    delimiter = ',',
    quotechar = '"',
    doublequote = True,
    skipinitialspace = True,
    lineterminator = '\r\n',
    quoting = csv.QUOTE_MINIMAL)

def writecsv(csvname,contents):
    f = open(csvname, 'a', newline='')
    wr = csv.writer(f)
    wr.writerow(contents)
    f.close()
    
CLASSES=('car','bus','truck', 'special vehicle', 'motorcycle','bicycle','personal mobility','person','Traffic_light', 'Traffic_sign')

ls = glob('2DBB/test/images','*')


for idx,filename in enumerate(ls[:1000]):
        
    output = inference_detector(model, filename)
    for cidx, oos in enumerate(output):
        if len(oos)>0:
            for pos in oos:
                #파일명, 예측 클래스, xmin, ymin, xmax, ymax, confidence score
                writecsv('vis_ver2/detection_box_preds.csv', [filename,CLASSES[cidx],pos[0],pos[1],pos[2],pos[3],pos[4]])
    out = model.show_result(filename,output)
    
    
    cv2.imwrite('vis_ver2/visualize_test/'+filename.split('/')[-1],out)

        