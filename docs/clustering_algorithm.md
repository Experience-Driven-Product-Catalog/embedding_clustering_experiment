# 군집화·대표명 선정 계약

## 1. 군집 입력 단위

원천 행을 직접 군집화하지 않는다. 각 단계에서 문자열을 중복 제거해 노드를 만든다.

- B: `unique(raw_attribute)`
- D-aspect: `unique(raw_aspect)`
- D-status: `unique(aspect_cluster_id, raw_status)` 중 non-null status

모니터 단일 카테고리이므로 B와 D-aspect에는 boundary가 없다. D-status의 `aspect_cluster_id`는 aspect/status 계층을 보존하는 유일한 boundary다.

각 노드는 `source_row_count`, `unique_review_count`를 보유한다. 두 빈도는 embedding이나 linkage에 가중치로 넣지 않는다.

## 2. 임베딩과 군집 형성

동일 로컬 KR-SBERT 모델로 고유 문자열을 L2 정규화해 임베딩한다. cosine distance와 complete linkage를 사용하며 단계별 절단 거리는 다음과 같다.

| 단계 | 거리 임계값 |
| --- | ---: |
| B | 0.22 |
| D-aspect | 0.3591 |
| D-status | 0.20 |

complete linkage이므로 한 군집의 모든 구성원 쌍의 최대 cosine distance가 해당 임계값 이하여야 한다. 실행 중 이를 다시 계산해 검증한다.

D-status에서는 설정된 반대 상태 쌍과 명시적 부정 쌍의 사전 계산 거리를 2로 바꾼 뒤 같은 complete-linkage 절차를 적용한다.

## 3. 실제 관측 대표명

크기 `n`인 군집에서 각 구성원 `i`의 다른 구성원까지 평균 거리 `D_i`를 계산한다. 평균 거리가 가장 작은 실제 구성원이 Medoid다. 동률이면 문자열 오름차순으로 결정한다.

중심 후보 수는 다음과 같다.

- `n=1`: 1개
- `n=2`: 2개
- `3<=n<=9`: 3개
- `n>=10`: 5개

평균 거리 순으로 중심 후보를 제한한 뒤 aspect/status 역할 필터를 적용한다. 남은 후보에 대해 다음 점수를 계산한다.

```text
C_i = clip(1 - D_i / 2, 0, 1)
F_i = log(1 + review_frequency_i) / max_j log(1 + review_frequency_j)
S_i = 0.70 * C_i + 0.30 * F_i
```

`S_i`가 가장 높은 후보를 대표명으로 선택한다. 동률 정렬은 평균 거리, 고유 리뷰 지지도 내림차순, 문자열 오름차순이다. 대표명은 항상 구성원 중 하나이며 새 문자열을 생성하지 않는다.

singleton은 원래 문자열을 그대로 상속한다. 대표명 gate를 통과하지 못한 multi-member 군집은 원천 행 매핑에서 해당 raw 값을 그대로 유지한다.

## 4. null과 제외 규칙

`raw_status` null은 유효한 원천 상태다. status 군집 입력에서는 제외하고 C/D 최종 `status`에서 같은 행에 null로 보존한다.

`raw_aspect == "전반적 상품 경험"`인 Opinion Unit은 기존 실험과 동일하게 C/D 입력·지표에서 제외한다. 제외 행 수는 manifest와 자동 무결성 검사에 기록한다.

## 5. 행 단위 정규화 계보

각 군집 노드는 `cluster_id`, `canonical_label`, `naming_status`와 함께 `mapping_applied`, `mapping_distance`를 보유한다. 이 네 값은 원천 행으로 many-to-one 조인되어 B의 attribute 계보와 D의 aspect/status 계보로 저장된다.

`mapping_applied`는 선택된 대표명과 raw 문자열이 실제로 다를 때만 true다. `mapping_distance`는 raw 노드 임베딩에서 실제 최종 대표명 임베딩까지의 순수 cosine distance다. D-status 군집 형성에 사용한 opposition/negation 보정 거리는 계보 거리에 섞지 않는다. 대표명 gate가 실패하면 최종값은 raw 값이므로 `mapping_applied=false`, `mapping_distance=0`으로 기록하고 gate 결과는 `naming_status`로 보존한다. null `raw_status`는 status 군집 노드가 없으므로 status 계보 전체를 null로 유지한다.

B/D 결과의 모든 행에는 같은 실행의 `embedding_model_id`, `normalization_version`, `normalization_run_id`, `normalization_config_sha256`도 기록한다. 설정 hash는 결과 디렉터리에 복사된 config의 정규화 관련 부분을 식별하고, run ID는 입력 hash와 산출물 목록이 있는 `run_manifest.json`으로 연결한다.
