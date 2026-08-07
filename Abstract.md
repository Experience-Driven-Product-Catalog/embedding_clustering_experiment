# 모니터 속성 임베딩 군집화 실험

모니터 리뷰에서 추출한 두 속성 테이블을 사용해 구조화 여부와 군집 정규화 여부의 효과를 A–D 2×2로 비교한다. 군집화와 자동 평가는 항상 **입력 Parquet 전체**를 사용하며 sample 입력·sample 실행 옵션이 없다. 사람에게 제시하는 리뷰·군집 평가 태스크만 `config.yaml`의 N과 seed로 재현 가능하게 추출한다.

## 변경된 데이터 계약

상품, 리뷰, 속성은 서로 다른 테이블이라는 전제를 따른다. 이 프로젝트는 DuckDB를 만들거나 product/review 테이블을 복제하지 않는다. 속성 테이블에는 외부 연결키로 `review_idx`만 사용한다. `idx`는 각 속성 행의 기존 논리 PK이므로 그대로 보존한다.

Representative Attribute 입력의 정확한 열은 다음과 같다.

```text
idx, review_idx, raw_attribute, sentiment
```

Opinion Unit 입력의 정확한 열은 다음과 같다.

```text
idx, review_idx, raw_aspect, raw_status, excerpt, opinion, sentiment
```

- `sentiment`는 `positive`, `negative`, `mixed`, `neutral`, `unknown` 중 하나인 단일 열이다. 파생 sentiment 열을 만들지 않는다.
- `raw_status`는 근거 있는 상태를 특정할 수 없을 때만 null일 수 있다. C/D의 `status`도 동일한 행에서만 null을 보존한다.
- `product`, `product_name`, `productName`, `product_category`, `review`가 속성 입력에 포함되면 이전 통합 스키마를 잘못 사용한 것으로 간주해 실행을 중단한다.

## A–D 실험

| 실험 | 입력 | 군집 정규화 | 결과에 추가되는 열 |
| --- | --- | --- | --- |
| A | Representative Attribute | 없음 | `attribute = raw_attribute` |
| B | Representative Attribute | 있음 | `attribute`와 attribute 정규화 계보 |
| C | Opinion Unit | 없음 | `aspect = raw_aspect`, `status = raw_status` |
| D | Opinion Unit | 있음 | 정규화한 `aspect`, `status`와 각 정규화 계보 |

각 결과는 입력 행의 모든 원래 열을 값과 dtype까지 그대로 유지하고 최종 열 및 필요한 계보 열만 추가한 Parquet이다. 결과에는 product/review/category 정보가 복제되지 않는다.

`raw_aspect == "전반적 상품 경험"`인 행은 기존 실험 계약과 동일하게 C/D에서 제외한다. A/B의 행은 모두 사용한다.

## 정규화 계보 계약

정규화를 수행하는 B와 D는 최종값만 저장하지 않고 각 원천 행이 어느 군집과 대표명을 거쳐 변환되었는지도 같은 결과 Parquet에 저장한다.

- B: `attribute_cluster_id`, `attribute_naming_status`, `attribute_mapping_applied`, `attribute_mapping_distance`
- D-aspect: `aspect_cluster_id`, `aspect_naming_status`, `aspect_mapping_applied`, `aspect_mapping_distance`
- D-status: `status_cluster_id`, `status_naming_status`, `status_mapping_applied`, `status_mapping_distance`
- 공통 실행 계보: `embedding_model_id`, `normalization_version`, `normalization_run_id`, `normalization_config_sha256`

`*_mapping_applied`는 raw 값과 최종 저장값이 실제로 다른지를 나타낸다. `*_mapping_distance`는 raw 문자열 임베딩과 실제 저장된 대표명 임베딩 사이의 순수 cosine distance다. status cannot-link를 위해 보정한 거리는 이 값에 사용하지 않는다. 대표명 gate를 통과하지 못해 raw 값을 그대로 보존한 행은 거리 `0`, mapping false이며 구체적인 사유는 `*_naming_status`에 남는다. `raw_status`가 null이면 모든 status 계보 열도 null이다.

`normalization_version`은 `config.yaml`에서 관리하는 사람 판독용 정규화 계약 버전이다. `normalization_config_sha256`은 필터, embedding, clustering, 대표명 설정과 이 버전을 정렬된 JSON으로 직렬화한 SHA-256이다. `normalization_run_id`는 실행 결과 디렉터리명과 같으므로 해당 실행의 복사된 config와 manifest, 입력 파일 hash까지 추적할 수 있다. A/C는 정규화를 하지 않는 baseline이므로 군집 계보 열을 만들지 않는다.

## 군집화 계약

- 임베딩 모델: `snunlp/KR-SBERT-Medium-extended-klueNLItriplet_PARpair_QApair-klueSTS`
- 거리·linkage: cosine distance + complete linkage
- 임계값: B `0.22`, D-aspect `0.3591`, D-status `0.20`
- B는 전체 `unique(raw_attribute)`를 한 번에 군집화한다.
- D-aspect는 전체 `unique(raw_aspect)`를 한 번에 군집화한다.
- 모니터 단일 카테고리이므로 `product_category` hard boundary는 코드와 산출물에서 완전히 제거했다.
- D-status만 의미 계층을 보존하기 위해 `aspect_cluster_id` 내부에서 `unique(raw_status)`를 군집화한다. 이것은 product category 경계가 아니다.
- 문자열 빈도는 군집 형성에 가중치로 사용하지 않고, 실제 관측 대표명을 고를 때 고유 리뷰 빈도 보조 점수로만 사용한다.
- 대표명은 생성하지 않고 cluster 구성원 중 하나를 선택한다. 반대 상태와 명시적 부정은 기존 configurable cannot-link 규칙을 유지한다.

세부 알고리즘은 [docs/clustering_algorithm.md](docs/clustering_algorithm.md)에 정리했다.

## 실행

```bash
uv sync
uv run python run_experiment.py
```

다른 설정 파일을 사용하려면 다음처럼 실행한다. 모든 상대 경로는 해당 설정 파일의 디렉터리를 기준으로 해석한다.

```bash
uv run python run_experiment.py --config /path/to/config.yaml
```

단일 설치형 명령도 동일한 진입점을 사용한다.

```bash
uv run embedding-clustering-experiment
```

`--sample`, `--create-samples` 옵션은 의도적으로 제공하지 않는다.

사람이 평가할 태스크 수와 추출 seed는 한 곳에서 설정한다. `task_count` N은 리뷰 평가와 군집 평가에 각각 동일하게 적용된다.

```yaml
evaluation:
  user_evaluation:
    task_count: 100
    random_seed: 17171771
```

## 결과 구조

실행별 결과는 `experiment_results/YYYYMMDD-HHMMSS/` 아래 새 폴더에 저장되며 기존 실행을 덮어쓰지 않는다.
따라서 계보 열은 이 변경 이후 새로 실행한 결과에만 생성되며, 이전 실행 Parquet은 소급 수정하지 않는다.

```text
config.yaml
experiment.log
run_manifest.json
results/
  experiment_a.parquet
  experiment_b.parquet
  experiment_c.parquet
  experiment_d.parquet
clustering/
  experiment_a_inventory.parquet
  experiment_b_nodes.parquet
  experiment_b_clusters.parquet
  experiment_c_inventory.parquet
  experiment_d_aspect_nodes.parquet
  experiment_d_aspect_clusters.parquet
  experiment_d_status_nodes.parquet
  experiment_d_status_clusters.parquet
evaluation/
  automatic_evaluation_summary.json
  data_integrity_checks.parquet
  experiment_metrics.parquet
  paired_experiment_metrics.parquet
  factorial_effects.parquet
  review_cohort_summary.parquet
  cluster_stage_metrics.parquet
  cluster_node_quality.parquet
  cluster_quality.parquet
  nearest_cluster_pairs.parquet
  risky_clusters.parquet
  review_normalization_changes.parquet
  user_evaluation_tasks.parquet
  cluster_evaluation_tasks.parquet
  human_results/
    completed_evaluators/
      <evaluator-uuid>.parquet
```

`run_manifest.json`은 입력과 모든 산출물의 SHA-256, 행 수, 실제 device, 모델 경로, 패키지 버전을 기록한다. `normalization` 항목에는 결과 행과 동일한 정규화 버전·실행 ID·설정 SHA-256·모델 ID 및 거리 정의를 기록한다. 전체 실험 입력을 뜻하는 `scope`는 `full_input_datasets`, `sampling_used`는 항상 `false`다. 사람 평가 태스크 표본은 별도로 `user_evaluation_sampling_used`, `user_evaluation_task_count`, `cluster_evaluation_task_count`, `user_evaluation_random_seed`에 기록한다.

자동 지표와 정규화 전후 비교는 전체 입력 행 및 전체 공통 `review_idx` cohort를 사용한다. 사람 평가용 리뷰 후보는 네 실험에 공통으로 존재하면서 A, B, C, D 표시 결과가 모두 하나 이상인 리뷰만 허용한다. 이 후보를 `review_idx`로 정렬한 뒤 설정한 seed로 정확히 N개를 비복원 추출해 `user_evaluation_tasks.parquet`에 저장한다. 파일에는 `review_idx`와 A–D 표시 결과만 있으며, 표시 결과는 Parquet/Pandas 버전 간 재독 안정성을 위해 JSON 문자열 배열로 저장한다.

군집 평가 후보는 B, D-aspect, D-status의 대표명이 정해진 비-singleton 군집을 하나의 정렬된 pool로 합친다. 같은 seed로 정확히 N개를 비복원 추출해 `cluster_evaluation_tasks.parquet`에 저장한다. 리뷰 후보 또는 군집 후보가 N개보다 적으면 N을 임의로 줄이지 않고 실험을 명시적으로 중단한다.

## 간소화한 사용자 평가

사용자 평가는 React 빌드와 FastAPI 서버 대신 Streamlit 파일 하나로 제공한다. 사용자 평가는 전체 자동 실험의 완료 조건에 포함되지 않는다.

리뷰 본문은 속성 결과에 저장하지 않는다. 앱을 시작할 때 별도 review 테이블을 `review_idx` 또는 `idx`로 메모리에서만 조인한다.

```bash
uv sync --extra user-evaluation
uv run --extra user-evaluation user_evaluation.py \
  --tasks experiment_results/YYYYMMDD-HHMMSS/evaluation/user_evaluation_tasks.parquet \
  --reviews /path/to/reviews.parquet \
  --output-dir experiment_results/YYYYMMDD-HHMMSS/evaluation/human_results
```

`user_evaluation.py`를 직접 실행하면 같은 Python 환경의 Streamlit 서버를 자동으로 시작한다.

`--cluster-tasks`를 생략하면 `--tasks`와 같은 디렉터리의 `cluster_evaluation_tasks.parquet`을 자동으로 사용한다. 다른 위치의 파일을 사용할 때만 `--cluster-tasks /path/to/cluster_evaluation_tasks.parquet`을 지정한다.

앱은 N개 리뷰 평가를 먼저 제시한 뒤 N개 군집 평가를 제시한다. 점수는 각 기준의 1~5 버튼으로 선택한다. 속성명 집합이 정확히 같은 리뷰 결과끼리는 기준별
최초 선택 점수만 함께 반영되며, 이후 어느 쪽을 수정해도 다른 결과에는 다시 전파되지 않는다.

review 테이블은 `review_idx` 또는 `idx`, 그리고 `review`를 가져야 한다. 선택적으로 `productName` 또는 `product_name`을 화면에만 표시한다. 완료 전 응답은 디스크에 쓰지 않고, 리뷰 N개와 군집 N개를 모두 완료했을 때만 `completed_evaluators/<uuid>.parquet`을 원자적으로 한 번 저장한다. 한 파일 안에서 `evaluation_level`이 `review` 또는 `cluster`를 구분하며, 저장 결과에는 리뷰 본문이나 상품 정보가 포함되지 않는다.

현재 전체 실행 `20260803-213339`에는 평가자 3명의 완료 파일이 있다. 보고서 생성기는 이 파일을 자동 발견해 리뷰 A–D 평점·preference, 2×2 주효과, ICC(2,1), 군집 평점과 표본 coverage를 다시 계산한다. 실행 종료 당시의 자동 요약에 사용자 평가가 미완료로 남아 있어도 완료 파일 검증 결과를 우선한다.

## 검증

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run python -m py_compile run_experiment.py user_evaluation.py src/embedding_clustering_experiment/*.py
uv lock --check
```

테스트는 작은 synthetic fixture로 알고리즘 불변조건과 사람 평가 태스크의 정확한 N, seed 재현성, A–D 비어 있지 않음, 군집 비-singleton 조건을 검증한다. 실제 군집화와 자동 평가는 항상 전체 입력 데이터로 수행한다. 실행 중에는 직렬화 후 Parquet을 다시 읽어 원본 열 보존, 행·PK coverage, status null 보존, product category 경계 부재, complete-linkage 최대 거리, 관측 대표명 조건을 다시 검사한다.

## AI 쇼핑 Agent용 보고서

`generate_agent_reports.py`는 완료된 한 run의 Experiment D, 별도 review 테이블, 완료 사용자 평가를 결합해 LLM 호출 없이 두 종류의 산출물을 만든다. Agent용 JSON은 전체 수치·근거·검증 계보를 보존하고, 사람 검토용 Markdown은 각 표에 계산 기반 핵심 해설 1–2문장을 붙인다. 집계·점수·불확실성·근거 선택 계약은 [docs/agent_report_contract.md](docs/agent_report_contract.md)에 정의했다.

한 상품의 정적 카탈로그 분석 JSON과 표를 생성한다.

```bash
uv run python generate_agent_reports.py static \
  --run-dir experiment_results/YYYYMMDD-HHMMSS \
  --product-name "AOC 알파스캔 Q27G2S 게이밍 IPS 155 QHD 프리싱크 무결점" \
  --output-dir generated_reports/static
```

기본값은 run 아래 `evaluation/human_results`를 자동 탐색한다. 완료 파일을 다른 위치에서 관리할 때만 `--human-results-dir /path/to/human_results`를 추가한다.

정적 JSON의 `related_products`는 source를 제외하고 다음을 각각 최대 3개 반환한다.

- `similar_products`: 전체 canonical aspect의 4감성 profile을 category-prior smoothing(`τ=5`)한 experience similarity 순위. 표시 Top 10은 점수에 쓰지 않으며, exact aspect-status overlap은 설명 component로만 반환한다.
- `weakness_repair_alternatives`: source에서 support 2 이상인 부정 aspect-status를 exact cluster ID 기피 조건으로 바꾼 대안 순위. 약점 utility 75%, 경험 유사도 25%를 결합하며 근거가 없으면 빈 결과와 이유 code를 반환한다.

예시의 기본 정적 보고서는 `examples/generated/static_catalog_report.md`에, 고정 seed `20260804`로 선택한 다른 상품 `SMART 삼성전자 M7 S32BM700`의 동일 형식 샘플은 `examples/generated/random_product_smart_s32bm700/static_catalog_report.md`에 둔다.

기존 리뷰의 정규화된 Opinion Unit을 제출 리뷰로 사용해 동적 의사결정 제안 예시를 생성한다. `--review-idx`는 반복 지정할 수 있으므로 여러 리뷰를 하나의 제출로 함께 분석할 수 있다.

```bash
uv run python generate_agent_reports.py proposal \
  --run-dir experiment_results/YYYYMMDD-HHMMSS \
  --review-idx 50967 \
  --output-dir generated_reports/proposal
```

새 리뷰를 처리하는 upstream이 이미 추출·정규화한 submission JSON을 전달할 수도 있다. 이 생성기는 자유서술 리뷰를 LLM으로 추출하지 않으며, `excerpt` 연속성·감성값·현재 run의 canonical aspect/status와의 일치를 검증한다.

```bash
uv run agent-review-report proposal \
  --run-dir experiment_results/YYYYMMDD-HHMMSS \
  --submission-json /path/to/submission.json \
  --output-dir generated_reports/proposal
```

동적 출력 JSON은 제출 Opinion Unit, 같은 상품의 다른 리뷰와의 review-level 관계, 다른 리뷰에서 언급되지 않은 aspect-status, 그리고 제출된 부정 속성을 보완할 대안 3개의 근거·점수를 보존한다. 대안은 정적 보고서와 같은 전체 canonical aspect profile의 experience similarity와 weakness utility를 1:3으로 결합하며, 비언급은 상태 부재로 간주하지 않는다. 사용자 평가 파일이 일부만 있거나 계약을 위반하면 불완전 결과를 사용하지 않고 생성에 실패한다.


