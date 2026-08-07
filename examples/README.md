# 보고서 예시와 검증 관계

- `analysis_expected/`: 구현 전 설계한 한 상품·한 사용자 리뷰의 골든 출력 계약
- `generated/`: `generate_agent_reports.py`가 사용자 평가 완료 run `20260803-213339`에서 재생성한 출력
- `generated/random_product_smart_s32bm700/`: 고정 seed `20260804`로 기본 정적 예시와 다른 상품을 선택해 생성한 같은 형식의 정적 보고서
- `generated/multi_aspect_review_52282/`: 8개의 canonical aspect를 가진 SMART 삼성전자 M7 S32BM700 리뷰의 동적 제안서 예시
- `demo_request.json`: 외부 upstream이 넘길 수 있는 구조화된 review submission 예시
- `verification.json`: 양쪽 JSON/Markdown의 byte 동일성, 핵심 집계·제출 Opinion Unit·다른 리뷰 관계·대안 근거 및 사용자 평가 불변식 43개를 검증한 영수증

정적 예시는 `AOC 알파스캔 Q27G2S 게이밍 IPS 155 QHD 프리싱크 무결점`, 동적 예시는 review `50967`을 사용한다. 추가 정적 샘플은 `SMART 삼성전자 M7 S32BM700`이다. 8개 속성 동적 샘플은 review `52282`를 사용한다. 동적 JSON은 제출 리뷰의 raw/canonical Opinion Unit, 같은 상품의 다른 리뷰와의 관계, 비언급 pair, exact aspect-status가 없을 때의 동일 aspect status Top 3, 최대 3개 대안의 약점 보완 근거를 보존한다. 정적 JSON에는 완료 평가자 3명의 A–D 분석도 보존한다. 정적 Markdown은 빠른 확인용 bounded table view로, `Most Debated Aspect`의 원문 리뷰 샘플과 Related products의 두 표로 끝난다.
