# AI 쇼핑 Agent용 리뷰 분석 산출물 계약

## 1. 목적과 범위

이 프로젝트의 보고서는 자연어 요약문이 아니다. 다음 두 객체를 LLM 없이 결정적으로 생성한다.

1. `static_catalog_analysis`: 저장된 한 상품의 리뷰 관찰값, 분모, 불확실성, 근거, 정규화 계보를 제공한다.
2. `dynamic_review_decision_proposal`: 제출 리뷰의 구조화된 Opinion Unit을 카탈로그의 다른 리뷰와 비교하고, 제출된 부정 관찰을 보완할 대안 상품을 최대 3개 제공한다.

현재 데이터만으로 가격, 객관 스펙, 재고, 배송, 프로모션 또는 사용자의 장기 프로필을 판단하지 않는다. 새 자유서술 리뷰의 Opinion Unit 추출도 이 저장소의 책임이 아니다. 동적 입력은 이미 `aspect`, `status`, `sentiment`로 구조화되어야 하며, 가능하면 현재 정규화 run의 cluster ID를 포함한다.

## 2. 현재 데이터와 신뢰 경계

2026-08-03 전체 데이터 실행 `20260803-213339`과 이후 완료된 사용자 평가에서 확인한 범위는 다음과 같다. 자동 요약은 실행 종료 시점의 스냅샷이므로, 사용자 평가 완료 여부는 `human_results/completed_evaluators/*.parquet`을 다시 검증한 결과를 우선한다.

| 항목 | 값 |
|---|---:|
| 상품 수 | 9 |
| 원본 리뷰 수 | 749 |
| Opinion Unit이 있는 리뷰 | 748 |
| 원본 Opinion Unit | 2,583 |
| `전반적 상품 경험` 제외 후 D Opinion Unit | 2,401 |
| D aspect 문자열 / 군집 | 870 / 408 |
| D `(aspect, status)` 원시 조합 / 최종 조합 | 1,600 / 1,320 |
| 자동 무결성 검사 | 38 / 38 통과 |
| 완료 평가자 | 3명 |
| 사용자 평가 과제 | 리뷰 100개 + 군집 100개 / 평가자 |
| 사용자 평가 행 | 리뷰 평점 1,200개 + 군집 평점 300개 |

상품·리뷰·속성은 별도 테이블이다. D 결과의 `review_idx`를 review 테이블의 `idx`에 `many_to_one`으로 조인하고, 조인되지 않은 Opinion Unit이 있으면 생성에 실패한다.

## 3. 공통 집계 계약

### 3.1 기본 분모

한 리뷰가 같은 aspect를 여러 번 언급해도 상품의 대표 의견을 과대 계상하지 않도록 다음 두 vote grain을 만든다.

- aspect vote: `(product_name, review_idx, aspect_cluster_id)`
- aspect-status vote: `(product_name, review_idx, aspect_cluster_id, status_cluster_id)`

`opinion_unit_count`는 원시 관찰량으로 별도 보존하고, sentiment 비율의 기본 분모는 review-level vote 수다.

### 3.2 한 리뷰 안의 sentiment 결합

같은 vote grain에 여러 Opinion Unit이 있으면 다음 규칙으로 하나의 sentiment를 만든다.

```text
known = observed_sentiments - {unknown}

known이 비어 있음       -> unknown
known의 고유값이 1개    -> 그 sentiment
known의 고유값이 2개 이상 -> mixed
```

따라서 각 표의 `positive + negative + mixed + neutral + unknown`은 항상 `supporting_review_count`와 같다.

### 3.3 비율과 불확실성

- `mention_rate = supporting_review_count / catalog_review_count`
- sentiment share의 분모는 해당 aspect 또는 pair의 `supporting_review_count`
- positive와 negative share에는 Wilson score 95% 구간을 제공한다.
- `normalized_entropy`는 5개 sentiment 분포의 Shannon entropy를 `log2(5)`로 나눈 값이다. 0은 한 sentiment에 완전히 집중, 1은 균등 분포다.
- 비언급은 상태의 부재가 아니라 `NO_EVIDENCE`다.

### 3.4 근거 선택

각 aspect-status-sentiment에서 최대 N개 excerpt를 다음 순서로 결정적으로 선택한다.

1. 서로 다른 `review_idx`
2. aspect/status mapping distance 합이 작은 행
3. 원천 `idx` 오름차순

근거에는 `review_idx`, `excerpt`, `opinion`, raw/final label, cluster ID, mapping distance를 보존한다. `sentiment`는 원시 Opinion Unit의 polarity이고 `review_vote_sentiment`는 해당 근거가 지지하는 표의 review-level vote 버킷이다. 둘이 다른 경우에도 숨기지 않는다. 근거가 없는 sentiment에는 빈 배열을 반환한다.

## 4. 정적 카탈로그 분석 보고서

### 4.1 최상위 구조

```text
schema_version
report_type = static_catalog_analysis
report_id
source
human_evaluation
product
coverage
normalization_reduction
aggregation_contract
sentiment_distribution
aspect_summary[]
aspect_status_summary[]
most_debated_aspect
related_products
normalization_quality
quality_flags[]
agent_capabilities
```

`report_id`에는 상품명뿐 아니라 집계 설정, 사용자 평가 상태, 완료 결과 파일 집합 SHA-256의 canonical JSON hash를 포함한다. 같은 run과 상품이라도 분모·근거 수·정밀도 또는 사후 사용자 평가가 달라져 내용이 바뀌면 ID도 달라진다. `aggregation_contract`에는 두 vote grain, sentiment 결합 규칙, 분모, 근거 수, 반올림 정밀도를 기계 판독 가능한 값으로 포함한다.

### 4.2 `aspect_summary[]`

각 행은 다음 정보를 가진다.

- `rank`, `aspect_cluster_id`, `aspect`
- `supporting_review_count`, `mention_rate`
- 5개 sentiment count/share
- `dominant_sentiment`, `dominant_share`
- positive/negative Wilson 95% 구간
- `normalized_entropy`
- product 내부 mapping 적용률과 distance 분위수
- 전역 `risky_clusters.parquet`과 연결한 정규화 위험

### 4.3 `aspect_status_summary[]`

aspect 행의 필드에 `status_cluster_id`, `status`와 sentiment별 실제 근거를 추가한다. 모든 pair를 지원 수 내림차순으로 보존하므로, 자연어 top-N 요약에서 사라질 수 있는 소수 의견도 Agent가 조회할 수 있다.

### 4.4 정규화 감소와 Most Debated Aspect

`normalization_reduction`은 이 상품에 실제로 등장한 D Opinion Unit에서 다음 세 grain의 원시 문자열 수와 최종 canonical cluster 수를 함께 반환한다. 감소율의 분모는 각각의 원시 고유값 수이며, 원시값이 없으면 감소율은 `null`이다.

- aspect: `unique(raw_aspect)` → `unique(aspect_cluster_id)`
- status: 같은 aspect 안의 `unique(aspect_cluster_id, raw_status)` → `unique(status_cluster_id)`; null status는 제외
- pair: `unique(raw_aspect, raw_status)` → `unique(aspect_cluster_id, status_cluster_id)`; null status는 명시적 null pair로 보존

`most_debated_aspect`는 표시용 aspect Top 10 중 `abs(positive_share - negative_share)`가 가장 작은 행이다. 최소 한 개의 positive 또는 negative review vote가 있어야 하며, 동률은 support, label, cluster ID 순으로 결정한다. 이 객체는 positive·negative별 실제 Opinion Unit 근거를 최대 3개씩 보존하고, 각 근거의 `review_text`로 Markdown에 원문 review를 표시한다. 조건을 만족하는 aspect가 없으면 `null`이다.

### 4.5 Related products

`related_products`는 source 상품 자신을 제외한 카탈로그 상품을 최대 3개씩 두 관점으로 반환한다. 표시용 Top 10 표를 점수에 사용하지 않고, **모든 canonical aspect**의 product-review-aspect vote를 사용한다. 따라서 표에서 보이지 않는 희소 aspect도 유사도와 대안 계산에서 사라지지 않는다.

#### 유사 상품

aspect `a`, 상품 `p`에 대해 `m(p,a) = supporting_review_count / catalog_review_count`, `r(p,a) = n(p,a) / (n(p,a) + 5)`, `q(p,a) = 1 - unknown_share(p,a)`를 둔다. positive/negative/mixed/neutral 네 채널의 sentiment 확률은 해당 aspect의 카테고리 prior로 `τ = 5`만큼 smoothing한다.

```text
v(p,a,k) = sqrt(m(p,a)) * r(p,a) * q(p,a) * smoothed_sentiment_probability(p,a,k)
experience_similarity(p,q)
  = cosine(v(p), v(q)) * sqrt(weighted_aspect_overlap(p,q))
weighted_aspect_overlap(p,q)
  = sum_a min(m(p,a), m(q,a)) / sum_a max(m(p,a), m(q,a))
```

- 9개 상품 MVP에서는 불안정한 희귀도 가중을 피하기 위해 `idf = 1`이다.
- `unknown`은 경험 감성 채널에는 넣지 않고 `q`로 품질을 감점한다.
- exact `(aspect_cluster_id, status_cluster_id)` mention-rate overlap은 별도 component로 반환하지만 score에는 합치지 않는다. status 임베딩의 근접성으로 `있음/없음` 같은 반대 상태를 같은 것으로 처리하지 않는다.
- JSON은 `experience_similarity`, aspect mention/sentiment similarity, evidence overlap, shared support reliability, exact aspect-status overlap, 공통 aspect와 정규화 위험 코드를 보존한다. 정렬은 score 내림차순, 상품명 오름차순으로 결정한다.

#### 관찰된 약점을 보완하는 대안

source의 aspect-status pair 중 negative review vote가 있고 support가 2 이상이며 status cluster ID가 있는 항목만 기피 requirement로 쓴다. 요구의 status match는 **같은 status cluster ID만** 허용한다. 각 후보의 약점 utility는 기존 동적 의사결정 점수(`[-1, 1]`)이며, 서로 다른 의미의 두 순위를 섞지 않도록 다음처럼 결합한다.

```text
weakness_repair_score
  = 0.25 * experience_similarity
  + 0.75 * ((weakness_utility_score + 1) / 2)
```

실제 근거가 확인된 requirement가 하나도 없는 후보는 제외하고 최대 3개를 반환한다. support 2 이상의 source 부정 pair가 없으면 `NO_HIGH_SUPPORT_NEGATIVE_ASPECT_STATUS_EVIDENCE`, 후보 근거가 없으면 `NO_CANDIDATE_WITH_REPAIR_EVIDENCE`를 반환하며 빈 순위를 억지로 채우지 않는다.

### 4.6 Markdown 표시 계약

정적 Markdown은 다음 순서로 렌더링한다.

1. bold 처리한 상품명, 카탈로그 리뷰 수, D Opinion Unit 수와 세 정규화 감소율
2. `속성 감성 행렬 (상위 10개)`: `reviews(mention rate)`, positive/negative count(rate), mixed/neutral/unknown count
3. `속성-상태 행렬 (상위 10개)`: 위 정보와 positive Wilson lower
4. `가장 논쟁적인 속성`: 양쪽 비율 차이, `### 긍정 리뷰 샘플`, `### 부정 리뷰 샘플` 아래의 원문 review sample 최대 3개씩
5. `관련 상품`: `### 유사 상품`에서 시작하며, 유사 상품 3개의 experience similarity/evidence overlap/support reliability와 약점 보완 대안 3개의 weakness utility/experience similarity/repair score 표

정적 Markdown의 모든 product name은 bold로 표시하며, 독자가 보는 field label의 underscore는 공백으로 바꾼다. JSON field name과 schema는 snake_case를 유지한다. 각 표에는 분모·점수 범위를 밝히는 고정 설명과 집계값으로 채운 핵심 문장을 인접하게 둔다. 정적 Markdown은 `Related products`의 약점 보완 표에서 끝나며, `Human evaluation`과 `Quality flags`는 Agent용 JSON에만 보존한다. Markdown은 사람이 빠르게 검토하는 bounded view일 뿐이고, Agent는 JSON의 전체 행과 근거를 사용한다.

### 4.7 필수 품질 플래그

- 전체 리뷰 대비 Opinion Unit 보유 리뷰 coverage
- `전반적 상품 경험` 제외 수
- review timestamp 부재
- 가격·스펙·재고·배송 데이터 부재
- 사용자 평가 미완료 여부
- 완료 평가가 포괄한 D 위험 군집 수와 최소 평가자 수 한계
- 상품에 등장한 위험 aspect/status 군집

## 5. 동적 의사결정 제안서

### 5.1 입력과 검증 경계

동적 제안서는 사용자가 제출한 리뷰를 요구조건으로 재해석하지 않는다. 제출 리뷰의 Opinion Unit 자체를 관찰값으로 보존하고, 그 관찰값을 같은 상품의 카탈로그 리뷰 및 대안 상품 근거와 비교한다. 새 자유서술 리뷰의 추출·정규화는 upstream 책임이며, 이 생성기는 LLM을 호출하지 않는다.

```json
{
  "submission_id": "demo-review-50967",
  "submitted_at_local": "20260803-213400",
  "product_name": "...",
  "reviews": [
    {
      "source_review_idx": 50967,
      "review": "...",
      "opinion_units": [
        {
          "raw_aspect": "화면 크기",
          "aspect": "화면 크기",
          "raw_status": "작게 느껴짐",
          "status": "작아 보임",
          "excerpt": "...",
          "opinion": "...",
          "sentiment": "negative",
          "aspect_cluster_id": "D-A-000392",
          "status_cluster_id": "D-S-001196"
        }
      ]
    }
  ],
  "excluded_products": ["..."]
}
```

`--review-idx`는 현재 D 결과에서 위 구조를 결정적으로 만들며 여러 번 지정해 여러 리뷰를 제출할 수 있다. 외부 submission은 canonical label이 유일하면 cluster ID를 생략할 수 있지만, 모호하거나 카탈로그에 없는 label은 생성 전에 실패한다. `excerpt`는 반드시 해당 review 원문의 연속 부분 문자열이고, 배송·배달·포장·서비스·사은품 등 비상품 속성은 거부한다. `전반적 상품 경험`만 status 없이 허용하며 카탈로그 pair 비교에서는 제외한다. `source_review_idx`가 있으면 product와 원문이 카탈로그 행과 정확히 일치해야 한다.

원본 상품은 어떤 submission에서도 대안 후보에서 제외한다. `proposal_id`는 정규화된 submission, report 설정, ranking 계약 및 실행 manifest hash의 canonical JSON hash를 포함한다.

### 5.2 다른 리뷰와의 관계

각 제출 Opinion Unit을 `(aspect_cluster_id, status_cluster_id)`로 묶어, 같은 상품의 전체 review-level vote와 비교한다. `catalog_review_count`, `aspect_review_count`, `same_status_review_count`는 전체 카탈로그 기준이고, 제출 리뷰가 카탈로그에서 왔으면 그 review index를 제외한 `other_status_sentiment`가 비교 기준이다. 따라서 제출자가 이미 본 같은 리뷰를 다수 의견으로 다시 세지 않는다. 다른 리뷰에 동일한 canonical aspect-status가 없으면 `other_aspect_top_statuses`에 해당 aspect의 다른 status를 review support 내림차순으로 최대 3개 반환한다. 이 표는 aspect 전체 언급 수와 동일 status의 비교 분모를 혼동하지 않도록, exact pair가 없다는 사실을 먼저 명시한 뒤 제공한다.

제출 감성과 다른 리뷰의 dominant sentiment가 모두 positive/negative이며 같으면 `ALIGNS_WITH_OTHER_REVIEW_MAJORITY` (`일치합니다`), 다르면 `CONTRADICTS_OTHER_REVIEW_MAJORITY` (`일치하지 않습니다`)다. support가 없거나 mixed/neutral/unknown이면 `INSUFFICIENT_OR_NON_DIRECTIONAL_OTHER_REVIEW_EVIDENCE` 또는 `NOT_MENTIONED_BY_OTHER_REVIEWS` (`판단하기 어렵습니다`)로 기록한다. 상태가 없는 Unit은 aspect 수준에서만 비교하고, 다른 리뷰에 없는 non-null pair는 `unmentioned_aspect_status`에 보존한다.

### 5.3 대안 상품 추천

제출 리뷰에서 canonical status를 가진 negative aspect-status만 약점 보완 조건으로 쓴다. 후보의 리뷰 경험 유사도는 정적 카탈로그 보고서와 동일하게 전체 canonical aspect의 four-sentiment profile, category shrinkage (`τ=5`), weighted aspect overlap으로 계산한다. source의 표시 Top 10은 후보 점수에 쓰지 않는다.

조건 `c`와 후보 `p`에 대해 같은 status cluster의 부정 근거와, 같은 aspect 안에서 다른 status cluster의 positive 근거를 각각 다음처럼 보수적으로 계산한다.

```text
negative_evidence_strength(p, c)
  = support_reliability * negative_wilson_upper_95

positive_alternative_strength(p, c)
  = max(support_reliability * positive_wilson_lower_95)
    over same-aspect, different-status, positive-dominant pairs

condition_utility(p, c)
  = positive_alternative_strength - negative_evidence_strength

weakness_utility(p)
  = submitted review vote 수를 가중치로 한 condition_utility 평균

weakness_repair_score(p)
  = 0.25 * experience_similarity(p)
  + 0.75 * ((weakness_utility(p) + 1) / 2)
```

같은 status cluster만 원하지 않는 상태의 증거로 사용하며, near-status match는 허용하지 않는다. 조건에 실제 카탈로그 근거가 하나도 없는 후보는 제외하고 `weakness_repair_score desc`, `weakness_utility desc`, `experience_similarity desc`, product name 순으로 최대 3개를 반환한다. 부정 pair가 없으면 `NO_NEGATIVE_SUBMITTED_ASPECT_STATUS`, 후보 근거가 없으면 `NO_CANDIDATE_WITH_REPAIR_EVIDENCE`를 반환한다. 이 점수는 구매 우열이나 구매 확률이 아니라 제출 리뷰에서 관찰한 약점을 유사한 맥락에서 보완할 근거의 정렬값이다.

### 5.4 Markdown 표시 계약

Markdown은 `동적 의사결정 제안서`, display-math 형식으로 분리한 제출 review 원문, Opinion Unit 표와 고정 field 정의, `다른 리뷰와의 관계`, `언급되지 않는 aspect-status`, `대안 상품 추천` 순서만 사용한다. Opinion Unit 표의 열은 `raw_aspect`, `aspect`, `raw_status`, `status`, `excerpt`, `opinion`, `sentiment`이며, 제출 상품과 표의 대안 상품명은 bold 처리한다. 관계 section에서는 aspect를 항상 bold 처리하고, exact aspect-status가 다른 리뷰에 없으면 해당 사실과 다른 status Top 3 표를 표시한다. 대안 section에는 weakness utility·experience similarity·weakness repair score의 3:1 규칙과 1순위 상품이 가장 높은 최종 점수로 추천됐다는 계산 기반 문장을 둔다. legacy `Request`, `Decision`, `Candidate ranking`, `Human evaluation`, `Quality flags` section은 렌더링하지 않는다.

## 6. 사용자 평가 통합 계약

### 6.1 완료 판정과 계보

실행 시점의 `automatic_evaluation_summary.json`은 사용자 평가 전 스냅샷이므로 완료 판정의 source of truth가 아니다. 기본 경로 `evaluation/human_results/completed_evaluators/` 또는 명시한 `--human-results-dir`의 Parquet을 매번 읽고 다음 조건을 모두 만족해야 `human_evaluation.status = completed`로 기록한다.

- 결과 파일마다 파일명과 같은 단일 evaluator UUID 및 단일 완료 시각
- 리뷰·군집 평가 과제 ID와 정확히 같은 cohort
- 평가자×리뷰마다 A–D가 각각 한 번, preference가 정확히 하나
- 평가자×군집마다 평점이 정확히 한 번
- 모든 Likert 평점이 정수 1–5, level별 전용 열 외에는 null
- 평가자 최소 3명

완료 파일은 manifest 사후 산출물이므로 개별 SHA-256과 정렬된 파일 hash 집합의 SHA-256을 별도로 기록한다. 외부 산출물에는 원본 evaluator UUID 대신 `session_N` alias만 저장한다. 파일이 없으면 `not_available`로 보고서를 생성하되 경고하고, 파일이 일부 있거나 잘못되었으면 불완전 결과를 묵인하지 않고 생성을 중단한다.

### 6.2 리뷰 평가 분석

평가자 점수를 먼저 `(review_idx, experiment)`에서 평균한 뒤 리뷰 100개를 통계 단위로 사용한다. 평균과 표준편차 외에 고정 seed로 리뷰를 5,000회 복원 추출한 95% bootstrap 구간을 제공한다. 평가자 행 300개를 독립 표본처럼 취급하지 않으므로 같은 리뷰의 반복 평가로 구간이 과도하게 좁아지는 것을 피한다.

2×2 효과는 다음과 같이 계산한다.

```text
structured representation = ((C + D) - (A + B)) / 2
clustering                = ((B + D) - (A + C)) / 2
interaction               = (D - C) - (B - A)
```

완료 결과에서 구조화 표현의 평균 주효과는 사실 충실도 `+0.210`, 핵심 설명력 `+1.445`, 정보 포괄성 `+1.815`점이며 세 bootstrap 구간의 하한이 모두 0보다 크다. 클러스터링 주효과 세 구간은 모두 0을 포함한다. A/B/C/D preference는 각각 `4/2/113/181`회로, D가 평가자-리뷰 선택 300회 중 60.3%를 차지한다.

### 6.3 군집 평가 분석과 한계

군집 cohesion과 canonical label fit도 1–5점, 군집 task 단위 평균과 bootstrap 구간으로 집계한다. 평가자 일치도는 리뷰와 군집 모두 ICC(2,1), 즉 two-way random-effects absolute-agreement single-measure로 제공한다.

표본 군집의 평균 평점은 4.683–4.896점이고 4점 이상 비율은 95.2–99.0%다. 그러나 5점 비율이 79.8%여서 ceiling risk가 있고, 비-singleton 군집 coverage는 B 22.5%, D-aspect 19.4%, D-status 25.0%다. 더 중요한 bounded risky cluster coverage는 B `0/10`, D-aspect `0/10`, D-status `4/10`이므로 높은 표본 평점을 모든 군집의 품질 보증으로 일반화하지 않는다. 평가자 수도 사전 최소치인 3명에 정확히 머물러 agreement와 일반화 정밀도가 제한된다.

### 6.4 JSON과 Markdown의 역할

Agent가 소비하는 정적 JSON의 `human_evaluation`에는 검증 영수증, 실험별 분포, preference, paired contrast, factorial effect, ICC, 군집 표본·위험 coverage와 기계 판독 가능한 conclusion code를 모두 보존한다. Markdown은 bounded inspection view이며 각 표 앞뒤에 해당 표의 분모·의미와 가장 중요한 신호를 계산식 기반의 한국어 1–2문장으로 붙인다. 이 문장은 고정 template과 집계값으로만 생성하며 LLM을 호출하지 않는다.

## 7. 선행 연구에서 반영한 원칙

- Hu and Liu의 feature-based review summary는 상품 feature별 의견 수와 실제 review sentence를 함께 제시했다. 현재 계약은 이를 5개 sentiment, review-level 분모, 정규화 계보로 확장한다. DOI: <https://doi.org/10.1145/1014052.1014073>
- SemEval-2014 Task 4는 aspect target/category와 polarity를 구분해 평가한다. 현재 계약도 aspect와 sentiment를 섞지 않고 status를 별도 관찰 상태로 보존한다. <https://aclanthology.org/S14-2004/>
- Bar-Haim et al.은 aspect sentiment 수치만으로는 설명 근거와 상충 의견을 충분히 표현하지 못한다고 지적한다. 따라서 분포·entropy와 실제 excerpt를 함께 제공한다. <https://aclanthology.org/2021.acl-long.262/>
- Wayfair의 대규모 시스템은 aspect-sentiment pair를 통합하고 빈번한 aspect와 대표 review를 선택한다. 현재 구현은 자연어 생성 단계를 제거하고 그 이전의 구조화 결과를 직접 산출물로 사용한다. <https://aclanthology.org/2025.emnlp-industry.31/>
- 소표본 proportion은 단순 비율만으로 과신하기 쉬우므로 Wilson score interval을 함께 제공한다. <https://doi.org/10.1080/01621459.1927.10502953>

## 8. 구현 및 검증 기준

- JSON 직렬화는 key와 행 정렬이 결정적이어야 한다.
- 같은 run·입력·설정으로 생성한 결과는 byte-identical해야 한다.
- 입력 D의 모든 `review_idx`가 review table에 조인되어야 한다.
- 선택한 Opinion Unit 입력과 사용한 run artifact는 manifest의 SHA-256 및 byte 수와 일치해야 한다.
- D 결과의 원천 열은 manifest에 기록된 Opinion Unit 입력과 값·dtype이 일치해야 한다.
- 5개 sentiment count 합은 모든 aspect/pair에서 support 수와 같아야 한다.
- share 합은 support가 있는 행에서 반올림 오차 범위 내 1이어야 한다.
- Wilson 구간과 entropy는 `[0, 1]`이어야 한다.
- evidence excerpt는 원본 review의 연속 부분 문자열이어야 한다.
- evidence의 `review_vote_sentiment`는 evidence가 속한 sentiment 표 버킷과 같아야 한다.
- `normalization_reduction`의 normalized 수는 해당 상품 D 결과의 실제 canonical ID cardinality와 같아야 한다.
- `most_debated_aspect`는 표시 Top 10 안에서 positive/negative share 차이가 최소인 aspect여야 하며, 표본은 해당 sentiment review vote를 지지해야 한다.
- related similarity는 source 상품을 후보로 포함하지 않고, 모든 canonical aspect를 사용하며, 최대 3개만 score 내림차순·상품명 오름차순으로 반환해야 한다.
- related exact aspect-status overlap은 동일 canonical ID pair만 세고 experience similarity 또는 weakness utility의 근접 status match로 사용하지 않아야 한다.
- 약점 보완 requirement는 source negative support 2 이상과 non-null status ID를 만족해야 하며, 대안은 실제 근거 coverage가 0인 후보를 포함하지 않아야 한다.
- 동적 submission의 excerpt는 해당 review 원문의 연속 부분 문자열이어야 하며, 제출 product와 cluster ID/label이 현재 카탈로그와 일치해야 한다.
- 동적 관계 비교는 같은 상품의 review-level vote를 사용하고, source_review_idx로 제출한 리뷰는 다른 리뷰 분포에서 제외해야 한다.
- 동적 대안은 submitted negative aspect-status만 조건으로 사용하고, source product를 제외하며, exact status ID의 부정 근거 및 same-aspect/different-status positive 근거를 분리해야 한다.
- 동적 대안 정렬은 `weakness_repair_score desc`, `weakness_utility desc`, `experience_similarity desc`, `product_name asc`로 고정한다.
- 사용자 평가 완료 파일은 schema, level별 null 계약, 과제 coverage, 중복, preference 개수와 최소 평가자 수를 fail-closed로 검증한다.
- 사용자 평가 평균·bootstrap·factorial effect·ICC·군집 및 위험 표본 coverage는 같은 입력과 seed에서 결정적이어야 한다.
- Markdown의 모든 표에는 분모 또는 해석 범위와 핵심 신호를 설명하는 계산 기반 문장이 인접해야 한다.
- 예시 산출물은 독립적으로 먼저 만든 뒤 구현 코드 출력과 구조·값을 비교한다.
