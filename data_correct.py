### DRG_~~~ 파일중 261개가 DRR로 되어있기 때문에 이를 바꾸어주기 위한 코드
import pandas as pd
import cv2
import os
import json
import argparse


#### visualize용  ####
coloc_dict = { 'car':(255,0,0)
               , 'bus':(0,255,0)
               , 'truck':(0,0,255)
               , 'special vehicle':(255,0,255)
               , 'motorcycle':(204,255,255)
               , 'bicycle':(153,51,0)
               , 'personal mobility':(204,255,0)
               , 'person':(51,102,255)
               , 'Traffic_light':(102,0,255)
               , 'Traffic_sign':(255,255,204)
}

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--csvpath', type=str)
parser.add_argument('--targetpath', type=str)

args = parser.parse_args()



csv = pd.read_csv(args.csvpath, header=None)

print(csv)


for label,inm in csv.values:
    inm = inm.replace('\\','/')
    if os.path.isfile(args.targetpath+inm.replace('DRG','DRR')):
        os.rename(args.targetpath+inm.replace('DRG','DRR'), args.targetpath+inm)

        
### data와 라벨 visualize해서 확인용 ###
    with open(label, 'r') as file:
        data = json.load(file)
    img = cv2.imread('2DBB/test/images/'+inm)
    if os.path.isfile('2DBB/visualize_test/' + data['image_name']) == False:
        for anno in data['Annotation']:
            cv2.rectangle(img, (int(anno['data'][0]), int(anno['data'][1])),
                          (int(anno['data'][2]), int(anno['data'][3])), coloc_dict[anno['class_name']], thickness=2)

        cv2.imwrite('2DBB/visualize_test/' + data['image_name'], img)
