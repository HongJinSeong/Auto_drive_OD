자율주행 관련 Object detection 대회 2위

라벨과 실제 이미지 파일명 다른 것을 확인하여 data correction 진행

Train 데이터의 양 자체는 80,000장이지만 동영상의 프레임이 분할 된 것으로 이루어져있고, 같은 영상 데이터의 이름을 기준으로 group by 하면 2,232개로 데이터의 다양성이 부족하여 4가지 데이터 증강 적용
- Multi-scale training -> 다양한 사이즈의 Object에 강건한 학습
- Photometric distortion -> 데이터에 주간, 야간, 심한 빛 반사 등 다양한 경우가 존재하여 밝기, 대비, 색조, 채도등에 변화를 주는 데이터 증강 방법 사용
- Random crop, Random horizontal flip

주어진 학습데이터에 class간에 불균형이 심하여 이를 해소하기 위한 class balanced sampling 진행하여 이미지별 class 등장 빈도 기반으로 sampling하여 기존 80,000장 대비 95,202장으로 증가함

Detection model은 UniverseNet을 사용하였고, COCO dataset을 통하여 pretrained weight를 활용하여 fine-tuning 진행

