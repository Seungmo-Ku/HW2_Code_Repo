# 분할 단위별 최적 n-gram 차수: LLM 생성 한국어 탐지 (CSE4057 HW2 코드)

어절 / 형태소 / 음절 단위로 n = 1–8 의 TF-IDF n-gram + 로지스틱 회귀 탐지기를 학습해,
단위마다 최적 n 이 어떻게 달라지는지 측정한다.

## 환경
- Python 3.12, `pip install -r requirements.txt`
- 생성(`generate.py`)만 API 키가 필요: `.env.example` 을 `.env` 로 복사해 키를 채움 (`.env` 는 제출물에 포함하지 않음)

## 데이터
| 파일 | 내용 | 포함 여부 |
|---|---|---|
| `data/sampled.jsonl` | 사람 글 1,000개 (AI Hub 논문요약 공학, 1992–2020) | **미포함** (AI Hub 이용 약관상 재배포 불가) — `sample_data.py` 로 재생성 |
| `data/gen_*.jsonl` | 생성기별 LLM 글 (Haiku 999, Gemini 998, GPT 1,000) | 포함 |
| KatFish 초록 | `abstract.jsonl` (Park et al., 2025) | 미포함 — KatFishNet 저장소의 `katfish_dataset/abstract.jsonl` |

AI Hub 「논문자료 요약」 데이터를 받아 `../018.논문자료 요약 데이터/...` 에 두거나 `--data_dir` 로 경로 지정.
KatFish 경로는 `cross_dataset.py --katfish <경로>` 로 지정 (기본: `../HW1_Code/katfishnet/katfish_dataset/abstract.jsonl`).

## 실행 순서 (논문의 모든 숫자 재현)
`make_tables.py` 가 `logs/` 의 일부 로그를 읽으므로, 아래처럼 로그 파일 이름을 그대로 써야 표가 만들어진다.
```bash
python sample_data.py --n 1000 --out data/sampled.jsonl                  # 사람 글 샘플링 (seed 42)
python generate.py --model anthropic:claude-haiku-5-5 --limit 1000        # LLM 글 생성 (이미 생성된 id 는 건너뜀, API 키 필요)
python generate.py --model openai:gpt-5.4-mini-2026-03-17 --limit 1000
python generate.py --model gemini:gemini-3.5-flash-lite --limit 1000

python experiment.py --out results/main.csv            > logs/full_main.log            # 본 실험
python experiment.py --mask --out results/mask.csv     > logs/full_mask.log            # 형식 단서 제거
for t in 25 50 100 200; do
  python experiment.py --mask --train_size $t --out results/mask_tr$t.csv > logs/full_mask_tr$t.log   # 학습량 곡선
done
python katfishnet_baseline.py        > logs/full_katfish.log                            # KatFishNet 쉼표 베이스라인
python katfishnet_baseline.py --mask > logs/full_katfish_mask.log
python cross_dataset.py --out results/cross.csv             > logs/full_cross.log      # 교차 데이터셋 (우리 <-> KatFish)
python cross_dataset.py --mask --out results/cross_mask.csv > logs/full_cross_mask.log
python analysis.py        > logs/analysis.log                                           # 상위 특징 + 부트스트랩 신뢰구간
python analysis.py --mask > logs/analysis_mask.log
python examples.py        > logs/examples.log                                           # 무작위 예시 + 오분류 사례
python examples.py --mask > logs/examples_mask.log
python sparsity.py        > logs/sparsity.log                                           # 어절 5-gram 희소성
python make_figures.py; python make_tables.py                                          # figures/*.pdf, results/tables.tex
```
LLM 생성은 API 응답이 매번 달라 정확히 같은 글이 나오지 않으므로, 결과 재현은 포함된 `data/gen_*.jsonl` 사용을 권장.
생성 이후 단계는 seed (분할 42, 43, 44 / 예시 추출 0) 고정으로 결정적이며, GPU 와 API 키가 필요 없다.
실행 시간 (MacBook Air M3, CPU): 본 실험 약 2분, 학습량 곡선 각 1분 이내, 생성 이후 전체 약 20분.
`make_tables.py` 를 다시 실행하면 `results/tables.tex` 의 캡션이 `TODO` 로 덮어써진다 (논문에 쓴 캡션은 이 저장소의 `tables.tex` 에 있음).

## 논문 표·그림 ↔ 재현 명령
| 논문 | 내용 | 만드는 명령 | 원자료 |
|---|---|---|---|
| Table 2 (`tab:main`), Figure 1 (`fig:main`) | 단위 × n 의 unseen AUROC | `experiment.py` → `make_tables.py`, `make_figures.py` | `results/main.csv` |
| Table 3 (`tab:summary`) | 단위별 최적 n·AUROC (원본/마스킹) + 쉼표 베이스라인 | `experiment.py [--mask]`, `katfishnet_baseline.py [--mask]` → `make_tables.py` | `results/main.csv`, `mask.csv`, `logs/full_katfish*.log` |
| Figure 2 (`fig:learning`) | 학습 문서 수별 곡선 | `experiment.py --mask --train_size …` → `make_figures.py` | `results/mask_tr*.csv`, `mask.csv` |
| §5.1 희소성 수치 | 어절 5-gram 특징 수·포함 글 비율 | `sparsity.py` | `logs/sparsity.log` |
| Table 4 (`tab:cross`), Figure 3 (`fig:cross`) | 교차 데이터셋 | `cross_dataset.py [--mask]` → `make_tables.py`, `make_figures.py` | `results/cross*.csv` |
| Table 5 (`tab:features`) | 단위별 상위 특징 | `analysis.py --mask` | `logs/analysis_mask.log` |
| §5.4 오류 예시, 부록 `tab:examples` | 무작위 예시·오분류 통계 | `examples.py` | `results/examples.json`, `logs/examples.log` |
| 부록 `tab:bootstrap` | 부트스트랩 95% 신뢰구간 | `analysis.py [--mask]` → `make_tables.py` | `logs/analysis*.log` |
| 부록 `tab:generators` | 생성기 쌍별 AUROC | `experiment.py` → `make_tables.py` | `results/main.csv` |

표·그림 번호는 컴파일 결과에 따라 달라질 수 있으니 label (`tab:…`, `fig:…`) 기준으로 볼 것.

## 파일
| 파일 | 역할 |
|---|---|
| `sample_data.py` | AI Hub 에서 사람 글 샘플링 |
| `generate.py` | 제목 + 요약 -> 같은 길이의 논문 단락 생성 (3개 API) |
| `experiment.py` | 길이 맞춤, 마스킹, 단위 x n x seed x 학습 생성기 실험, 최적 n 계산 |
| `katfishnet_baseline.py` | KatFishNet 쉼표 특징 5개 재구현 (Kkma/kss 대신 Kiwi) |
| `cross_dataset.py` | 우리 데이터 <-> KatFish 초록 교차 평가 |
| `analysis.py` | 상위 특징, 부트스트랩 95% 신뢰구간 |
| `examples.py` | 무작위 예시와 오분류 사례 추출 (`results/examples*.json`, `logs/examples*.log`) |
| `sparsity.py` | §5.1 근거: 학습량별 어절 5-gram 특징 수와 그 특징을 포함한 테스트 글 비율 |
| `make_figures.py`, `make_tables.py` | 논문 그림 / 표 |
| `results/` | 실험별 seed·생성기 쌍 단위 AUROC (csv), `tables.tex` |
| `logs/` | 생성 / 실험 실행 로그 |

## 설계 요약
- 평가: 문서 단위 80/20 분할 (seed 3개). 생성기 하나로 학습 -> 학습에 안 쓴 두 생성기로 테스트한 AUROC 평균 (unseen)
- 최적 n: seed 평균 AUROC 가 최고값의 1 표준편차 (seed 3개의 모표준편차, `np.std`) 이내인 가장 작은 n
- 길이 맞춤: 짝지은 사람 글과 LLM 글을 모두 짧은 쪽 길이의 90% 에서 어절 경계로 자름
- 마스킹 (`--mask`): 괄호 내용 삭제, 영문 -> E, 숫자 -> 0, 떨어진 조사 붙임
- 제외 문서: Haiku 안전 필터 거부 1건, Gemini 무료 한도 초과 2건 -> 세 생성기 공통 997문서
