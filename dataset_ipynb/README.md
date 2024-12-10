
# 프로젝트 제목
GNN 기반 추천 시스템

## 개요
이 프로젝트는 GNN(그래프 신경망)을 활용하여 사용자 메타데이터와 아이템 간의 상호작용을 기반으로 맞춤형 추천을 생성하는 시스템입니다. PyTorch Geometric을 사용하여 그래프 기반 모델링을 구현하고, 사전 학습된 BERT 임베딩을 통해 사용자 및 아이템 특징을 통합합니다.

## 주요 기능
- 사전 학습된 BERT(KLUE BERT)를 사용한 사용자 및 아이템 임베딩 생성.
- 사용자-아이템 상호작용 그래프 생성.
- GNN 기반 모델을 통한 추천 학습 및 예측.
- 사용자 맞춤형 추천 제공.
- 사용자의 위치 기반 가장 가까운 경로 추천.

## 폴더 구조
```
.
├── evaluate.py          # 추천 평가 스크립트
├── main.py              # GNN 모델 학습 스크립트
├── model.py             # GNNRecommender 모델 정의
├── preprocessing.py     # 데이터 전처리 및 임베딩 생성 스크립트
├── utils.py             # 데이터 로딩 및 유틸리티 함수
├── dataset/             # 데이터셋 폴더
│   ├── user.csv         # 사용자 메타데이터
│   ├── item.csv         # 아이템 메타데이터
│   ├── preferences.csv  # 사용자-아이템 상호작용 데이터
│   └── preprocessed_data.pt  # 전처리된 데이터 파일
```

## 설치 방법

### 필수 패키지 설치
다음 명령어를 사용하여 Conda 환경에서 필요한 패키지를 설치하세요:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

`requirements.txt` 파일의 의존성을 설치하려면:
```bash
pip install -r requirements.txt
```

## 사용법

### 데이터 전처리
다음 명령어를 실행하여 전처리를 수행하고 데이터를 생성합니다:
```bash
python preprocessing.py
```
위 명령은 `dataset/preprocessed_data.pt` 파일을 생성합니다.

### 모델 학습
다음 명령어를 실행하여 GNN 모델을 학습시킵니다:
```bash
python main.py
```

### 평가 및 추천
새 사용자에 대한 추천을 생성하려면 다음 명령어를 실행하세요:
```bash
python evaluate.py --data_path dataset/preprocessed_data.pt                    --model_path best_model.pth                    --item_csv dataset/item.csv                    --user_csv dataset/new_user.csv
```

### 결과
- **Top-N 추천:** 사용자에게 추천된 아이템 목록.
- **가장 가까운 경로:** 사용자의 위치를 기준으로 가장 가까운 산책로 목록.

## 주요 구성 요소

### 주요 컴포넌트
1. **모델 (`model.py`):**
   - `GNNRecommender`: 사용자-아이템 상호작용을 모델링하기 위한 GNN 모델.
2. **유틸리티 (`utils.py`):**
   - `create_hetero_data`: 데이터를 PyTorch Geometric의 `HeteroData` 형식으로 변환.
   - `load_preprocessed_data`: 전처리된 데이터를 `.pt` 파일에서 로드.
   - `EarlyStopping`: 검증 손실을 모니터링하여 과적합 방지.
3. **전처리 (`preprocessing.py`):**
   - KLUE BERT를 사용하여 사용자 및 아이템 메타데이터 임베딩 생성.
   - 상호작용 데이터를 학습, 검증, 테스트 데이터셋으로 분할.
4. **평가 (`evaluate.py`):**
   - 새 사용자 데이터를 임베딩.
   - 새 사용자에 대한 추천 및 가장 가까운 경로 제공.

## 데이터셋
- **User CSV:** 사용자 메타데이터 (예: 성별, 키, 선호 장소 등).
- **Item CSV:** 아이템 메타데이터 (예: 행정구역명, 경로 난이도 등).
- **Preferences CSV:** 사용자와 아이템 간의 상호작용 데이터.


