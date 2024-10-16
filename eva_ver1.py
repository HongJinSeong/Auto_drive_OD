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
import pandas as pd
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


    def load_annotations(self, ann_file):
        
        CLASSES_dict = {'car' : 0 , 'bus' : 1, 'truck' : 2, 'special vehicle' : 3, 'motorcycle' : 4,'bicycle' : 5 ,'personal mobility' : 6 
                        ,'person' : 7 ,'Traffic_light' : 8, 'Traffic_sign' : 9}
        
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        
        data_infos = []
        
        ls = pd.read_csv(ann_file, header = None)
        
        for idx,an in enumerate(ls.values):
            an=an[0]
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


print(f'Config:\n{cfg.pretty_text}')


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
    samples_per_gpu=10,
    workers_per_gpu=12,
    train=dict(
        type=cfg.dataset_type,
        ann_file='2DBB/new_train.csv',
        pipeline=cfg.train_pipeline),
     val=dict(
        type=cfg.dataset_type,
        ann_file='2DBB/new_valid.csv',
        pipeline=cfg.test_pipeline),
    test=dict(
        type=cfg.dataset_type,
        ann_file='2DBB/new_test.csv',
        pipeline=cfg.test_pipeline))

cfg.model.bbox_head.num_classes=10

cfg.device='cuda'
cfg.work_dir = 'checkpoints_ver2'

cfg.log_config.interval = 8000 #iteration 단위

cfg.seed = 2024

set_random_seed(cfg.seed, deterministic=False)

cfg.workflow = [('train', 1), ('val',1)]

cfg.evaluation = dict(interval=1, metric='mAP')

cfg.load_from = 'universenet50_2008_fp16_4x4_mstrain_480_960_2x_coco_20200815_epoch_24-81356447.pth'
cfg.runner = dict(type='EpochBasedRunner', max_epochs=24)

cfg.model.test_cfg['score_thr']=0.05

cfg.gpu_ids = range(1)


print(f'Config:\n{cfg.pretty_text}')

from mmdet.apis import init_detector, inference_detector, show_result_pyplot

# Build the detector
#model = init_detector(config, checkpoint, device='cuda:0')
checkpoint='checkpoints_ver2/epoch_24.pth'
# model = init_detector(cfg,checkpoint, device='cpu')
model = init_detector(cfg, checkpoint, device='cuda:0')
model = model.eval()

ls = pd.read_csv('2DBB/new_valid.csv', header = None)

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

### iou 계산
def iou(box1, box2):
  '''Compute the Intersection-Over-Union of two given boxes.
  Args:
    box1: array of 4 elements [cx, cy, width, height].
    box2: same as above
  Returns:
    iou: a float number in range [0, 1]. iou of the two boxes.
  '''

  lr = min(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
      max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])
  if lr > 0:
    tb = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
        max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])
    if tb > 0:
      intersection = tb*lr
      union = box1[2]*box1[3]+box2[2]*box2[3]-intersection

      return intersection/union

  return 0

### 이미지에 대한 precision recall 측정
### 이미지에 대한 iOU 측정이후 threshold 기준으로 class까지 맞으면  TP case count
### precision = TP / model로 예측한 이미지별 detection 갯수
### recall = TP / 이미지별 Ground truth 갯수
### 개별 이미지에 대한 평가를 위해 진행

### 메모리 문제 땜시 끊어서... 다시 시작 이후 체크
### 773  / 1581  /  2346

for idx,an in enumerate(ls.values[2346:]):
        an=an[0]
        json_data = {}
        with open(an, "r") as json_file:
            json_data = json.load(json_file)
            
        ansplit = an.split('/')

        filename = ansplit[0] + '/' + ansplit[1] + '/' + 'images'+'/'+ json_data['image_name']
        
        GT_COUNT = len(json_data['Annotation']) ## recall 용

        output = inference_detector(model, filename)
        
        DET_COUNT = 0 ## precision 용
        TP = 0 
        for i,dets in enumerate(output):
            DET_COUNT += len(dets)
            
            for an in json_data['Annotation']:
                for det in dets:        
                    if an['class_name'] == CLASSES[i]: ### True positive 찾기위해 같은 class 로 예측했을 때만 confidence score는 cfg.model.test_cfg['score_thr']=xxx에서 미리 처리됨
                        
                        ## 현재 좌표가 (xmin,ymin,xmax,ymax) 로 되어있어서 iou 함수에 맞게 (x,y,width,height)로 넣어줌
                        val = iou([det[0],det[1],det[2]-det[0],det[3]-det[1]], [an['data'][0],an['data'][1],an['data'][2]-an['data'][0], an['data'][3]-an['data'][1]])
                        
                        ## 만약에 1개의 object detection point에 여러개의 True positive가 있으면 1개만 True로 치고 나머지는 틀린것으로 처리
                        ## https://github.com/rafaelpadilla/Object-Detection-Metrics/issues/46 참조
                        
                        if val > 0.5:
                            TP += 1
                            break
        
        # evals['file_nm'].append(filename)
        # evals['TP_CNT'].append(TP)
        # evals['RECALL'].append(TP/GT_COUNT)
        # evals['PRECISION'].append(TP/DET_COUNT)
        
        writecsv('vis_ver1/EVAL_PER_IMG.csv', [filename,TP,TP/GT_COUNT,TP/DET_COUNT ])
        
        if os.path.isdir('vis_ver1/imgs/'+json_data['image_name'])==False:
            img = cv2.imread(filename)

            out = model.show_result(filename,output,score_thr=0.0)

            concat_img = cv2.hconcat([img, out])
            cv2.imwrite('vis_ver1/imgs/'+json_data['image_name'], concat_img)
            
            del img, out, concat_img
        json_file.close()
        
# df = pd.DataFrame(evals)
# df.to_csv('vis_ver1/EVAL_PER_IMG.csv', index=False)