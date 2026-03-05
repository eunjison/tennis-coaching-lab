-------------------------------------------------------------------
🎾 Azure AI Foundry Fine-Tuning 실습 Lab
-------------------------------------------------------------------
무인 테니스 연습장 AI 코칭 시스템 구축

📋 Lab 개요
시나리오
무인 테니스 연습장을 운영하는 사장으로서, 고객이 30분 동안 연습하는 모습을 촬영한 영상을 AI가 분석하여 전문 코칭 피드백을 제공하는 시스템을 구축합니다.
학습 목표

Azure AI Foundry 환경을 설정하고 Foundation Model을 배포합니다.
프롬프트 엔지니어링으로 모델 성능을 개선합니다.
RAG(Retrieval-Augmented Generation)를 적용하여 테니스 전문 지식을 보강합니다.
각 단계별 성능을 정량적으로 비교·평가합니다.
Fine-tuning을 적용하여 도메인 특화 모델을 만듭니다.
최종 완성 애플리케이션을 배포합니다.

아키텍처 개요
┌─────────────────────────────────────────────────────────┐
│                    완성 애플리케이션                       │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ 영상 업로드 │───▶│ 프레임 추출    │───▶│ GPT-4.1-nano  │  │
│  │ (30분 영상) │    │ (ffmpeg)      │    │ (Vision+FT)   │  │
│  └──────────┘    └──────────────┘    └───────┬───────┘  │
│                                              │          │
│                  ┌──────────────┐    ┌───────▼───────┐  │
│                  │ RAG 검색      │───▶│ 코칭 리포트    │  │
│                  │ (AI Search)   │    │ 생성          │  │
│                  └──────────────┘    └───────────────┘  │
└─────────────────────────────────────────────────────────┘
모델 선택: GPT-4.1-nano
항목상세모델명gpt-4.1-nano-2025-04-14파라미터 규모OpenAI 시리즈 중 최소Vision 지원✅ (이미지 입력 가능)Fine-tuning 지원✅ (Azure AI Foundry Serverless)컨텍스트 윈도우1,000,000 토큰추론 비용 (Global)Input: $0.10 / Output: $0.40 (per 1M tokens)학습 비용 (Global)~$2.00 per 1M training tokens선택 이유최저 비용, Vision 지원, Fine-tuning 지원
사전 준비물

Azure 구독 (무료 체험 가능)
Python 3.10 이상
VS Code 또는 원하는 IDE
Azure CLI 설치 (az 명령어)

비용 예상
단계예상 비용Lab 1~3: 프롬프트 엔지니어링~$0.50Lab 4: RAG (AI Search S0)~$1.00/일Lab 5~6: 평가~$1.00Lab 7: Fine-tuning (학습)~$5~15Lab 8: 평가~$1.00Lab 9: 애플리케이션~$2.00합계약 $10~20 (AI Search 별도)

💡 Tip: Developer Tier 배포를 사용하면 호스팅 비용 없이 24시간 동안 Fine-tuned 모델을 테스트할 수 있습니다.

-------------------------------------------------------------------
01. 프로젝트 구조 및 파일 목록 (폴더별 정리)
-------------------------------------------------------------------
tennis-coaching-lab/
├── lab01_environment/           # Lab 1: 환경 설정
│   └── test_connection.py       # 연결 테스트
├── lab02_baseline/              # Lab 2: Baseline 평가
│   ├── baseline_test_data.py    # 테스트 데이터 (10개)
│   └── baseline_evaluation.py   # Baseline 평가
├── lab03_prompt_engineering/    # Lab 3: 프롬프트 엔지니어링
│   ├── prompts.py               # 시스템 프롬프트
│   └── prompt_engineering_test.py # 프롬프트 테스트
├── lab04_rag/                   # Lab 4: RAG 적용
│   ├── tennis_knowledge.py      # 테니스 지식 문서
│   ├── simple_rag.py           # 로컬 RAG 구현
│   ├── azure_rag.py            # Azure AI Search RAG
│   ├── rag_evaluation.py       # RAG 평가
│   └── rag_evaluation_simple_rag.py # 간단 RAG 평가
├── lab05_comparison/            # Lab 5: 단계별 비교
│   └── compare_results.py       # 비교 대시보드
├── lab06_analysis/              # Lab 6: 한계점 분석
│   └── limitation_analysis.py   # 한계점 분석
├── lab07_finetuning/            # Lab 7: Fine-tuning
│   ├── prepare_training_data.py # 학습 데이터 생성
│   ├── training_data.jsonl      # 학습 데이터 (40개)
│   ├── validation_data.jsonl    # 검증 데이터 (5개)
│   ├── run_finetuning.py        # Fine-tuning 실행
│   └── check_finetuning_status.py # 학습 상태 확인
├── lab08_evaluation/            # Lab 8: 최종 평가
│   ├── finetuned_evaluation.py  # Fine-tuned 평가
│   └── final_comparison.py      # 최종 비교 대시보드
├── lab09_application/           # Lab 9: 완성 애플리케이션
│   ├── frame_extractor.py       # 프레임 추출
│   ├── coaching_engine.py       # AI 코칭 엔진
│   └── app.py                   # 웹 애플리케이션
├── shared/                      # 공유 파일들
│   ├── .env                     # 환경 변수
│   ├── main.md                  # 프로젝트 가이드
│   ├── eval_baseline.json       # Baseline 결과
│   ├── eval_prompt_v1.json      # 프롬프트 V1 결과
│   ├── eval_prompt_v2.json      # 프롬프트 V2 결과
│   ├── eval_rag.json            # RAG 결과
│   └── eval_rag_hybrid.json     # RAG Hybrid 결과
└── .venv/                       # Python 가상환경

-------------------------------------------------------------------
02. 실행 순서 요약 (폴더별 정리)
-------------------------------------------------------------------
# Lab 1: 환경 설정
cd lab01_environment
pip install openai python-dotenv Pillow opencv-python-headless flask azure-search-documents
python test_connection.py

# Lab 2: Baseline
cd ../lab02_baseline
python baseline_evaluation.py

# Lab 3: 프롬프트 엔지니어링
cd ../lab03_prompt_engineering
python prompt_engineering_test.py

# Lab 4: RAG
cd ../lab04_rag
python simple_rag.py  # 로컬 RAG
python rag_evaluation.py  # RAG 평가

# Lab 5: 비교
cd ../lab05_comparison
python compare_results.py

# Lab 6: 한계 분석
cd ../lab06_analysis
python limitation_analysis.py

# Lab 7: Fine-tuning
cd ../lab07_finetuning
python prepare_training_data.py
python run_finetuning.py
python check_finetuning_status.py <JOB_ID>

# Lab 8: Fine-tuning 평가
cd ../lab08_evaluation
python finetuned_evaluation.py
python final_comparison.py

# Lab 9: 완성 애플리케이션
cd ../lab09_application
python app.py

-------------------------------------------------------------------
03. 상세 Lab 가이드
-------------------------------------------------------------------

Lab 1: Azure AI Foundry 환경 설정
목표
Azure AI Foundry에서 프로젝트를 생성하고, GPT-4.1-nano 모델을 배포합니다.
1.1 Azure AI Foundry Hub 및 프로젝트 생성

Azure AI Foundry 포털에 접속합니다.
+ Create project를 클릭합니다.
다음 설정으로 프로젝트를 생성합니다:

설정값Project nametennis-coaching-labHub새로 생성 또는 기존 Hub 선택RegionEast US 2 또는 Sweden Central (Fine-tuning 지원 리전)
1.2 GPT-4.1-nano 모델 배포

프로젝트에서 Models + endpoints > + Deploy model 을 선택합니다.
gpt-4.1-nano를 검색하여 선택합니다.
배포 설정:

설정값Deployment namegpt-41-nano-baseDeployment typeGlobal StandardRate limit기본값

Deploy를 클릭합니다.
배포 완료 후 Endpoint URL과 API Key를 메모합니다.

1.3 개발 환경 설정
bash# 프로젝트 폴더 생성
mkdir tennis-coaching-lab
cd tennis-coaching-lab

# Python 가상환경
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 필요 패키지 설치
pip install openai python-dotenv Pillow opencv-python-headless flask
pip install azure-search-documents azure-identity

# 환경 변수 설정
cat > .env << 'EOF'
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_DEPLOYMENT=gpt-41-nano-base
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Lab 4에서 추가 (RAG용)
AZURE_SEARCH_ENDPOINT=https://<your-search>.search.windows.net
AZURE_SEARCH_KEY=<your-search-key>
AZURE_SEARCH_INDEX=tennis-knowledge
EOF

1.4 연결 확인
test_connection.py 수행
✅ 체크포인트: "연결 성공!" 메시지와 테니스 관련 응답이 출력되면 Lab 1 완료입니다.

Lab 2: Foundation Model 기본 성능 테스트 (Baseline)
목표
Fine-tuning 전 기본 모델의 테니스 코칭 능력을 측정하여 Baseline 점수를 확보합니다.
2.1 테스트 데이터 준비
실제 동영상 대신, 테니스 폼을 설명하는 텍스트 시나리오와 샘플 이미지를 사용합니다.
# baseline_test_data.py

2.2 Baseline 테스트 실행
# baseline_evaluation.py
python baseline_evaluation.py

✅ 체크포인트: 10개 테스트 케이스의 평균 점수와 비용이 eval_baseline.json에 저장됩니다. 이 점수가 이후 비교의 기준선(Baseline)입니다.

💡 예상 Baseline 점수: 프롬프트 엔지니어링 없이는 보통 40~60점 범위가 예상됩니다. 모델이 테니스 지식은 있지만, 코칭 포맷과 세부 사항이 부족할 수 있습니다.


Lab 3: 프롬프트 엔지니어링 적용
목표
시스템 프롬프트를 설계하여 모델이 전문 테니스 코치처럼 응답하도록 만듭니다.
3.1 시스템 프롬프트 설계
# prompts.py

3.2 프롬프트 엔지니어링 적용 테스트
# prompt_engineering_test.py

python prompt_engineering_test.py
✅ 체크포인트: eval_prompt_v1.json과 eval_prompt_v2.json이 생성됩니다. Baseline 대비 점수 향상을 확인하세요.


Lab 4: RAG (Retrieval-Augmented Generation) 적용
목표
테니스 전문 지식 문서를 Azure AI Search에 인덱싱하고, RAG를 통해 모델에 전문 지식을 보강합니다.
4.1 테니스 지식 문서 준비
# tennis_knowledge.py


4.2 Azure AI Search 인덱스 생성
# azure_rag.py
💡 간단 대안 (Azure AI Search 없이): 소규모 데이터이므로 로컬 벡터 검색도 가능합니다.
# simple_rag.py (Azure AI Search 대신 로컬 RAG)

4.3 RAG 적용 테스트
python# rag_evaluation.py

python rag_evaluation.py
✅ 체크포인트: eval_rag.json이 생성됩니다. RAG 컨텍스트가 추가되어 토큰 사용량이 증가했지만, 품질이 향상되었는지 확인하세요.

Lab 5: 단계별 성능 비교 평가
목표
Baseline, 프롬프트 엔지니어링, RAG의 성능을 종합적으로 비교·분석합니다.
5.1 비교 대시보드 생성

python compare_results.py
✅ 체크포인트: 비교 대시보드에서 각 단계별 점수, 비용, 지연시간 차이를 확인합니다.


Lab 6: 프롬프트 엔지니어링 / RAG의 한계 분석
목표
프롬프트 엔지니어링과 RAG만으로는 해결하기 어려운 문제들을 구체적으로 식별합니다.
6.1 한계점 분석 스크립트

python limitation_analysis.py
✅ 체크포인트: 프롬프트 엔지니어링과 RAG의 구체적 한계를 확인하고, Fine-tuning의 필요성을 이해합니다.

Lab 7: Fine-tuning 데이터 준비 및 학습
목표
테니스 코칭 전문 데이터를 JSONL 형식으로 준비하고, Azure AI Foundry에서 GPT-4.1-nano를 Fine-tuning합니다.
7.1 학습 데이터 생성
Fine-tuning에는 최소 50개 이상의 고품질 학습 데이터가 필요합니다. 여기서는 테스트 데이터(10개)와 추가 학습 데이터(40개)를 합쳐 50개를 준비합니다.
python# prepare_training_data.py

python prepare_training_data.py
7.2 Azure AI Foundry에서 Fine-tuning 실행
방법 1: Azure AI Foundry 포털 (GUI)

Azure AI Foundry 포털에 접속합니다.
프로젝트에서 Fine-tuning > + Fine-tune model을 클릭합니다.
다음과 같이 설정합니다:

설정값Base modelgpt-4.1-nanoFine-tuning methodSupervised Fine-Tuning (SFT)Training datatraining_data.jsonl (업로드)Validation datavalidation_data.jsonl (업로드)Training typeGlobal Training (비용 절감)

Hyperparameters (기본값 권장):

설정값설명Epochs3자동 설정 또는 3 권장Batch sizeauto자동 최적화Learning rate multiplierauto자동 최적화

Submit을 클릭하고 학습 완료를 기다립니다 (보통 30분~2시간).

방법 2: Python SDK
# run_finetuning.py

# check_finetuning_status.py

7.3 Fine-tuned 모델 배포
학습 완료 후:

Azure AI Foundry 포털에서 Models + endpoints 로 이동합니다.
Fine-tuned 모델을 선택하고 Deploy를 클릭합니다.
배포 설정:

설정값Deployment namegpt-41-nano-tennis-coachDeployment typeDeveloper Tier (테스트용, 24시간) 또는 Global Standard

.env 파일에 Fine-tuned 배포명을 추가합니다:

AZURE_OPENAI_FT_DEPLOYMENT=gpt-41-nano-tennis-coach
✅ 체크포인트: Fine-tuned 모델이 배포되어 API 호출이 가능한 상태입니다.

Lab 8: Fine-tuning 성능 평가
목표
Fine-tuned 모델의 성능을 기존 방법들과 비교하고, 비용 효율성을 분석합니다.
8.1 Fine-tuned 모델 평가
# finetuned_evaluation.py

8.2 전체 비교 대시보드 (Final)
# final_comparison.py
python final_comparison.py
✅ 체크포인트: 5가지 방법의 점수, 비용, 지연시간이 모두 비교되고 ROI 분석이 완료됩니다.


Lab 9: 완성 애플리케이션 구축
목표
영상 업로드 → 프레임 추출 → AI 분석 → 코칭 리포트까지 자동화된 웹 애플리케이션을 구축합니다.
9.1 프레임 추출 모듈
# frame_extractor.py

9.2 AI 분석 엔진
# coaching_engine.py

9.3 웹 애플리케이션 (Flask)
# app.py
python app.py
✅ 체크포인트: 브라우저에서 http://localhost:5000에 접속하여 데모 분석을 실행할 수 있습니다.