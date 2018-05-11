# GPU training with Cloud Machine Learning Engine

구글 클라우드 상에서 GPU를 사용하기 위한 방법

## 1. Colaboratory
![tesla k80](https://cdn-images-1.medium.com/max/1600/1*Kbta9F_ZiRQmvETa-JkOSA.png)
### 특징 : 
* The GPU used in the backend is K80
* The 12-hour limit is for a continuous assignment of VM. It means we can use GPU compute even after the end of 12 hours by connecting to a different VM.
### 장점 : 무료
### 단점 : GPU사용에 제한이 있음(머신 대수, 사용시간)  
## 2. Google cloud Ml-engine
![structure](https://cloud.google.com/ml-engine/docs/images/dist-tf-datalab-arch.svg?authuser=0&hl=ko)
### 특징 : 
* 통합 
    * Google 서비스는 함께 작동하도록 설계되었습니다. 기능 처리는 Cloud Dataflow, 데이터 저장소는 Cloud Storage, 모델 생성은 Cloud Datalab과 함께 작동 가능
* HyperTune
    * 모델에 맞는 값을 직접 찾기 위해 많은 시간을 소비하는 대신 HyperTune 기능으로 사전 분포형 매개변수를 자동으로 조정해 성능이 더 높은 모델을 보다 빠르게 제작 가능
* 관리형 서비스
    * 인프라에 대해 걱정할 필요 없이 모델 개발과 예측에 집중할 수 있습니다. 관리형 서비스가 모든 리소스 프로비저닝과 모니터링을 자동화 가능
* 확장 가능한 서비스
    * CPU와 GPU를 지원하는 관리형 분산 학습 인프라를 사용해 데이터 크기나 유형에 관계없이 모델을 제작할 수 있습니다. 다수의 노드에서 학습시키거나 병렬로 여러 실험을 실행해 모델 개발 속도를 높일 수 있음
* 노트북 개발자 환경
    * Cloud Datalab에 통합된 친숙한 Jupyter 노트북 개발자 환경을 사용해 모델을 만들고 분석할 수 있음
    * Colaboratory로 대체 가능
* 이식 가능한 모델
    * 오픈소스 TensorFlow SDK를 사용해 샘플 데이터세트에서 로컬로 모델을 학습시키고 규모에 맞게 Google Cloud Platform을 학습에 사용할 수 있습니다. Cloud Machine Learning Engine으로 학습시킨 모델은 로컬 실행이나 모바일 통합을 위해 다운로드할 수 있습니다.
### 장점 : 환경셋팅 이후 손쉬운 사용과 대용량의 컴퓨팅 파워제공
### 단점 : 사용하기에 따라 비용이 많이 들 수 있음
### 기타 :
* pricing(https://cloud.google.com/ml-engine/docs/pricing)
* runtime version(https://cloud.google.com/ml-engine/docs/tensorflow/runtime-version-list)

# Cloud ml-engine 사용법
## Training your model with Cloud ML Engine
다음 단계에서는 모델 개발에서 Google Cloud Platform에 대한 Training 작업 관리에 이르기까지 모델 교육에 대해 설명함
1. Cloud ML Engine을 사용하기위한 요구 사항 및 모범 사례를 고려하여, Tensorflow computation graph 및 교육 application을 작성
2. 트레이너 애플리케이션 및 종속성을 패키지로 만들고 패키지를 Cloud ML 엔진 프로젝트가 액세스 할 수있는 Google 클라우드 저장소에 저장 (이 단계는 gcloud 명령 행 도구를 사용하면 단순화됨).
3. 트레이너를 실행하기 위해 Cloud ML Engine 작업을 구성하고 실행
4. 실행한 작업 모니터링

Cloud ML 엔진의 고급 기능을 사용할 수도 있음
* 그래픽 처리 장치 (GPU)를 사용하도록 트레이너를 구성
* 하이퍼 매개 변수 튜닝을 사용하여 정확성을 위해 모델의 구성을 자동화

마지막으로, 문제가 발생할 경우 훈련 업무 문제를 해결해야 할 수도 있음

## Packaging a Training Application
* 코드를 패키지화하고 클라우드에 배치하는 단계
* Cloud Machine Learning Engine으로 트레이너 애플리케이션을 실행하려면 먼저 코드 및 모든 의존성을 Google Cloud Platform 프로젝트에서 액세스 할 수있는 Google 클라우드 저장소 위치에 배치필요
* Cloud ML Engine의 패키징 요구 사항에 대한 자세한 설명은 교육 개념 페이지에서 확인 가능
### Before you begin
트레이너 코드를 패키징하는 것은 모델 교육 과정의 일부이며, application을 클라우드로 이동하기 전에 다음 단계를 완료해야함 : 
* TensorFlow로 트레이너 어플리케이션을 개발

또한 다음을 확인해야함 :
* PyPI를 통해, 트레이너가 의존하는 모든 Python 라이브러리를 확인필요
* Cloud ML 엔진을 사용한 교육은 사용 된 리소스에 대해 계정에 비용을 청구되기 때문에, 트레이너를 로컬에서 테스트 필요
### Packaging and uploading your code and dependencies
학습용 애플리케이션을 GCS로 가져 오는 방법은 다음 요인에 따라 달라짐
* gcloud 도구 (권장)를 사용하거나 자체 솔루션을 코딩합니까?
* 패키지를 수동으로 만들어야합니까?
* 사용중인 Cloud ML 엔진 런타임에 포함되지 않은 추가 종속성이 있습니까?

참고 : 트레이너를 패키징하고 업로드하는 gcloud 옵션은 트레이닝 작업 제출 명령에 내장되어 있음
#### Gather required information
트레이너를 패키징하기 위해 다음사항이 필요
* 패키지 경로(Package path) : gcloud 명령 행 도구를 사용하여 트레이너를 패키징하는 경우, 트레이너 소스 코드의 로컬 경로를 포함해야함
* 작업 디렉토리(Job directory) : 작업 출력을위한 루트 디렉토리(GCS)
* 종속성 경로(Dependency paths)
* 모듈 이름(Module name) : 트레이너의 메인 모듈의 이름(트레이너를 시작하기 위해 실행하는 파이썬 파일), 권장 트레이너 프로젝트 구조를 사용하는 경우 모듈 이름은 trainer.task 임
* 스테이징 버킷(staging bucket) : 트레이너가 작업을 실행하는 데 필요한 교육 인스턴스에 교육 서비스를 복사 할 수 있도록 트레이너가 준비된 Google Cloud Storage 위치
#### gcloud 도구를 사용하여 패키지 및 업로드 (권장)
트레이너를 패키징하고 의존성과 함께 업로드하는 가장 간단한 방법은 gcloud 도구를 사용하는 것
1. --package-path 플래그를 트레이너 애플리케이션의 루트 디렉토리에 대한 경로로 설정
2. 패키지의 네임 스페이스 점 표기법을 사용하여 --module-name 플래그를 응용 프로그램의 주 모듈 이름으로 설정 (예 : 주 모듈의 권장되는 경우 ... / my_application / trainer / task.py, 모듈 이름 trainer.task입니다.)
3. --staging-bucket 플래그를 교육 및 종속 패키지 준비 단계에서 사용할 클라우드 저장소 위치로 설정

편리한 사용을 위해 환경설정 변수로 셋팅
```buildoutcfg
TRAINER_PACKAGE_PATH="/path/to/your/application/sources"
MAIN_TRAINER_MODULE="trainer.task"
PACKAGE_STAGING_PATH="gs://your/chosen/staging/path"
```
자세한 설명은 다음 링크 참조 : [링크](https://cloud.google.com/ml-engine/docs/tensorflow/training-jobs)

##### Package path
##### Job name : 프로젝트 내에서 고유한 이름(일반적인 방법은 현재 날짜와 시간을 모델 이름에 추가)
예제 :
```buildoutcfg
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="census_$now"
```

##### Job directory : 학습 출력에 사용할 클라우드 저장소 위치(작업을 실행하는 동일한 지역의 버킷에 위치를 사용)
```buildoutcfg
JOB_DIR="gs://your/chosen/job/output/path"
```

##### ml-engine 학습 suit 예제
```buildoutcfg
gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR  \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --region us-central1 \
    -- \
    --trainer_arg_1 value_1 \
    ...
    --trainer_arg_n value_n
```

# Comparison: Colab GPU use vs. gcloud GPU use


### Conclusion
- We have 10 people in MoT team
- We spend $30 per person for 24 hour use in gcloud
- Assuming 5 people use gcloud once per week
- A person spends 24 hour per gcloud-use
- Then, we monthly spend 
```bash
5 people * 4 weeks * $30 = $600 per month
```
- A four-1080ti-GPU machine requires around $10 hundred to buy
- For 16.66 month less use, gcloud is cheaper.


## Information

| Items                     |   Colab GPU                   |   gcloud GPU  per hour                      |
| ------------------------  | :---------------------------: | :-----------------------------------------: |
| Time based price          |   Free (limited to 12 hours)  |   $1.3578 / hour (BASIC_GPU in Asia-pacific)|
| Training unit based price |   Free (Limited to 12 hours)  |   $2.5144 / hour (BASIC_GPU in Asia-pacific)|


**BASIC_GPU scale tier**
A single worker instance with a single NVIDIA Tesla K80 GPU.

## Google Cloud GPU Pricing Examples

**1) Time based Price**
```bash
(Price per hour / 60 ) * job duration in minites
```

- A case study
    - use 12 hours
    - BASIC_GPU scale tier in Asia-pacific
```bash 
Price for use = (1.3578 / 60) * 12 * 60 = $16.2936
```
 
 
**2) Training unit based price**
```bash
(training units * base price per hour / 60) * job duration in minutes
```

- A case study
    - use 12 hours
    - BASIC_GPU scale tier  in Asia-pacific
    - base price per hour = $0.49   
```bash
Price for use = (2.5144 * $0.49 / 60) * 12 * 60 = $14.7846
```


## Reference: 
- About Pricing: https://cloud.google.com/ml-engine/docs/pricing  
- About scale tier: https://cloud.google.com/ml-engine/docs/tensorflow/training-overview#scale_tier