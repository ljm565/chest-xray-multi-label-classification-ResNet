# Multi-label Diagnosis using Chest X-ray and ResNet
## 설명
[MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)의 mimic-cxr-2.0.0-negbio, mimic-cxr-2.0.0-metadata 데이터와 Chest X-ray 이미를 바탕으로 multi-label 진단을 예측합니다.
한 명의 환자당 lowest alphanumeric의 기준으로 하나의 study_id를 선택하고, 선택된 study_id에 대해 AP view의 chest X-ray 이미지를 확용합니다.
만약 하나의 study_id에 여러개의 AP X-ray 이미지가 있다면, lowest alphanumeric 기준으로 하나의 이미지를 선택합니다.
이 실험에서 사용된 데이터는 민감한 의료 데이터이므로, 인가된 사람만 접근할 수 있습니다.
따라서 실험에 사용한 데이터를 공개하지 않겠습니다.
그리고 본 프로젝트에 대한 자세한 설명은 [흉부 X-ray와 ResNet을 이용한 환자 상태 예측](https://ljm565.github.io/contents/chest-xray-resnet1.html)을 참고하시기 바랍니다.
<br><br><br>

## 모델 종류
* ### ResNet101
    ResNet101를 이용하여 AP X-ray 이미지를 가지고 14개의 multi-label 진단을 예측합니다.
<br><br><br>


## 사용 데이터
* [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
<br><br><br>


## 사용 방법
* ### 학습 방법
    학습을 시작하기 위한 argument는 4가지가 있습니다.<br>
    * [-d --device] {cpu, gpu}, **필수**: 학습을 cpu, gpu로 할건지 정하는 인자입니다.
    * [-m --mode] {train, test}, **필수**: 학습을 시작하려면 train, 최종 학습된 모델을 가지고 있어서 학습된 모델의 성능을 평가하고 싶으면 test 모드를 사용해야 합니다. test 모드를 사용할 경우, [-n, --name] 인자가 **필수**입니다.
    * [-c --cont] {1}, **선택**: 학습이 중간에 종료가 된 경우 다시 저장된 모델의 체크포인트 부분부터 학습을 시작할 수 있습니다. 이 인자를 사용할 경우 -m train 이어야 합니다. 
    * [-n --name] {name}, **선택**: 이 인자는 -c 1 혹은 -m test 경우 사용합니다.
    중간에 다시 불러서 학습을 할 경우 모델의 이름을 입력하고, test 할 경우에도 확인할 모델의 이름을 입력해주어야 합니다(최초 학습시 config.json에서 정한 모델의 이름의 폴더가 형성되고 그 폴더 내부에 모델 및 모델 파라미터가 json 파일로 형성 됩니다).<br><br>

    터미널 명령어 예시<br>
    * 최초 학습 시
        ```
        python3 main.py -d cpu -m train
        ```
    * 중간에 중단 된 모델 이어서 학습 시
        <br>주의사항: config.json을 수정해야하는 일이 발생 한다면 base_path/config.json이 아닌, base_path/model/{model_name}/{model_name}.json 파일을 수정해야 합니다.
        ```
        python3 main.py -d gpu -m train -c 1 -n {model_name}
        ```
    * 최종 학습 된 모델의 test set에 대한 성능 결과를 확인할 시
        <br>주의사항: config.json을 수정해야하는 일이 발생 한다면 base_path/config.json이 아닌, base_path/model/{model_name}/{model_name}.json 파일을 수정해야 수정사항이 반영됩니다.
        ```
        python3 main.py -d cpu -m test -n {model_name}
        ```
    <br><br>

* ### 모델 학습 조건 설정 (config.json)
    * **주의사항: 최초 학습 시 config.json이 사용되며, 이미 한 번 학습을 한 모델에 대하여 parameter를 바꾸고싶다면 base_path/model/{model_name}/{model_name}.json 파일을 수정해야 합니다.**
    * physio_id: 인가된 데이터 다운 ID. 
    * pwd: 인가된 데이터 다운 비밀번호.
    * base_path: 학습 관련 파일이 저장될 위치.
    * model_name: 학습 모델이 저장될 파일 이름 설정. 모델은 base_path/model/{model_name}/{model_name}.pt 로 저장.
    * loss_data_name: 학습 시 발생한 loss data를 저장하기 위한 이름 설정. base_path/loss/{loss_data_name}.pkl 파일로 저장. 내부에 중단된 학습을 다시 시작할 때, 학습 과정에 발생한 loss 데이터를 그릴 때 등 필요한 데이터를 dictionary 형태로 저장.
    * img_size: X-ray 이미지 데이터 전처리 크기.
    * label: Multi-label 개수(여기서는 14).
    * batch_size: batch size 지정.
    * epochs: 학습 epoch 설정.
    * lr: 학습 learning rate.
    * pos_wts: {0, 1} 중 선택. BCEWithLogitsLoss의 pos_weight 사용 여부.
    * early_stop_criterion: Test set의 최대 accuracy 내어준 학습 epoch 대비, 설정된 숫자만큼 epoch이 지나도 나아지지 않을 경우 학습 조기 종료.    
    <br><br><br>


## 결과
* ### Test Set 결과
    * pos_weight (X) 모델
        * AUROC (macro): 0.7191
        * AUROC (micro): 0.8360
        * AUPRC (macro): 0.2804
        * AUPRC (micro): 0.4764<br><br>

    
    * pos_weight (O) 모델
        * AUROC (macro): 0.7594
        * AUROC (micro): 0.8649
        * AUPRC (macro): 0.3091
        * AUPRC (micro): 0.5313

<br><br><br>