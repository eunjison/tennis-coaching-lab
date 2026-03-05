# 🎾 Azure AI Foundry Fine-Tuning 실습 Lab

무인 테니스 연습장 AI 코칭 시스템 구축 프로젝트

## 📋 프로젝트 개요

이 프로젝트는 Azure AI Foundry를 활용하여 테니스 코칭 AI 시스템을 구축하는 실습입니다. Foundation Model부터 프롬프트 엔지니어링, RAG, Fine-tuning까지 AI 개발의 전 과정을 경험할 수 있습니다.

### 학습 목표
- Azure AI Foundry 환경 설정 및 Foundation Model 배포
- 프롬프트 엔지니어링을 통한 모델 성능 개선
- RAG(Retrieval-Augmented Generation) 적용
- 각 단계별 성능 정량적 비교·평가
- Fine-tuning을 통한 도메인 특화 모델 개발
- 완성 애플리케이션 배포

### 아키텍처 개요
```
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
```

## 📁 프로젝트 구조

프로젝트는 Lab 단계별로 폴더가 구성되어 있어 체계적인 학습이 가능합니다:

```
tennis-coaching-lab/
├── lab01_environment/           # Lab 1: 환경 설정
├── lab02_baseline/              # Lab 2: Baseline 평가
├── lab03_prompt_engineering/    # Lab 3: 프롬프트 엔지니어링
├── lab04_rag/                   # Lab 4: RAG 적용
├── lab05_comparison/            # Lab 5: 단계별 비교
├── lab06_analysis/              # Lab 6: 한계점 분석
├── lab07_finetuning/            # Lab 7: Fine-tuning
├── lab08_evaluation/            # Lab 8: 최종 평가
├── lab09_application/           # Lab 9: 완성 애플리케이션
├── shared/                      # 공유 파일들 (.env, 결과 파일 등)
└── .venv/                       # Python 가상환경
```

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 프로젝트 클론 및 환경 설정
cd lab01_environment
pip install -r requirements.txt
python test_connection.py
```

### 2. 순차적 실행
```bash
# 각 Lab 폴더로 이동하며 순차적으로 실행
cd ../lab02_baseline && python baseline_evaluation.py
cd ../lab03_prompt_engineering && python prompt_engineering_test.py
# ... 계속
```

## 📊 모델 선택: GPT-4.1-nano

| 항목 | 상세 |
|------|------|
| 모델명 | gpt-4.1-nano-2025-04-14 |
| 파라미터 규모 | OpenAI 시리즈 중 최소 |
| Vision 지원 | ✅ (이미지 입력 가능) |
| Fine-tuning 지원 | ✅ (Azure AI Foundry Serverless) |
| 컨텍스트 윈도우 | 1,000,000 토큰 |
| 추론 비용 | Input: $0.10 / Output: $0.40 (per 1M tokens) |
| 학습 비용 | ~$2.00 per 1M training tokens |

## 💰 비용 예상

| 단계 | 예상 비용 | 설명 |
|------|----------|------|
| Lab 1~3 | ~$0.50 | 프롬프트 엔지니어링 |
| Lab 4 | ~$1.00/일 | RAG (AI Search S0) |
| Lab 5~6 | ~$1.00 | 평가 |
| Lab 7 | $5~15 | Fine-tuning (학습) |
| Lab 8 | ~$1.00 | 평가 |
| Lab 9 | ~$2.00 | 애플리케이션 |
| **합계** | **약 $10~20** | AI Search 별도 |

> 💡 **Tip**: Developer Tier 배포를 사용하면 호스팅 비용 없이 24시간 동안 Fine-tuned 모델을 테스트할 수 있습니다.

## 📋 사전 준비물

- ✅ Azure 구독 (무료 체험 가능)
- ✅ Python 3.10 이상
- ✅ VS Code 또는 원하는 IDE
- ✅ Azure CLI 설치 (`az` 명령어)

## 📖 상세 가이드

각 Lab의 상세한 실행 가이드와 설명은 `shared/main.md` 파일을 참고하세요.

## 🎯 성능 비교 결과

프로젝트 실행 후 예상되는 성능 향상:

| 방법 | 예상 점수 | 특징 |
|------|----------|------|
| Baseline | 40~60점 | 기본 모델 성능 |
| 프롬프트 V1 | 70~80점 | 구조화된 프롬프트 |
| 프롬프트 V2 | 80~90점 | Few-shot 예제 포함 |
| RAG | 85~95점 | 전문 지식 보강 |
| Fine-tuning | 90~100점 | 도메인 특화 학습 |

## 🤝 기여

이 프로젝트는 교육용으로 개발되었으며, Azure AI Foundry의 다양한 기능을 학습하는 데 초점을 맞추고 있습니다.

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 제공됩니다.
