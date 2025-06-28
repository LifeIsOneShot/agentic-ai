아래는 **Adaptive Planning** 관련 주요 논문들의 플래너 구조를 정리한 내용입니다. 각 논문이 제안하는 시스템 구성 요소와 플로우를 자세히 다뤘어요.

---

## 1. MACI (Multi-Agent Collaborative Intelligence)

**구조**: 3‑계층 모듈식 아키텍처 ([arxiv.org][1])

1. **Meta‑Planner**

   * 전체 목표와 제약을 분석 → 역할(role)과 종속성(dependency) 위주로 *의존성 그래프* 생성
   * 각 역할에 필요한 에이전트를 할당하고 역할 간 흐름을 설계 ([themoonlight.io][2])

2. **Agent Repository**

   * **공통 에이전트**: 제약 검증, 상식 통합, 논리 평가 등
   * **도메인‑특화 에이전트**: 전문 플래닝 알고리즘 적용
   * 입력/출력 인터페이스와 컨텍스트 윈도우 제한을 준수 ([themoonlight.io][2])

3. **Run‑Time Monitor**

   * 계획 실행 중 모니터링 → deviation 발생 시 재조정 트리거
   * 리소스, 역할 재할당 동시 수행 ([themoonlight.io][3], [themoonlight.io][2])

> 요약: MACI는 ‘메타‑플래너 → 역할별 에이전트 배치 → 실행 모니터링’의 순환 구조이며, 역할 기반 책임 분리와 validation을 통해 계획의 안정성과 확장성을 확보합니다.

---

## 2. ALAS (Adaptive LLM Agent System)

**구조**: 상태 추적 + 로컬 보상 기반의 모듈형 에이전트 구조 ([arxiv.org][4])

1. **Role Specification**

   * 플래닝 단계에서 *역할(role)* 정의 (예: 드라이버, 픽업 요청 처리자)
   * 실행 시 구체적 바인딩: `RolePickup(Jane, locA) → RoleToBePickedUp(Emily, locB)` ([researchgate.net][5])

2. **Agent Factory / Execution Layer**

   * 각 역할에 대해 에이전트 프로필 구성 (기능, 상태, 로그 구조, 보상 로직 포함)
   * 실패나 리소스 부족 시 *compensation logic* 실행 ([researchgate.net][5])

3. **Validator Agent**

   * 독립된 에이전트로 플랜 유효성, 시간 일관성, 보상 범위 검증
   * 검증 실패 시 보상 계획 적용 및 로컬 보정으로 전체 플랜 유지 ([themoonlight.io][2])

> 요약: ALAS는 역할 정의 → 역할 기반 실행 에이전트 구성 → 독립 검증 & 로컬 보상 루프를 통해 **충돌 없이 계획 복원력(resilience)** 을 보장합니다.

---

## 3. PlanGEN (구조 요약)

Bib 직접 찾기 어렵지만 Reddit 요약 정보 기준입니다:

* **제약 추출기**: 자연어 명세로부터 제약조건을 자동 추출
* **검증 에이전트**: 발생 가능한 실패 시나리오 감지
* **알고리즘 선택기**: 제약 구조에 따라 플래닝 전략 결정
* **적응 루프**: 실패 시 plan → verify → replan 3단계 순환 구조 ([arxiv.org][4])

> 요약: PlanGEN은 **제약 추출 → 검증 → 알고리즘 선택 → 실패 시 재계획**의 반복을 통해 자연언어 기반 플랜 적응성을 확보합니다.

---

## 4. ChronoPlanner (Reptile‑MADDPG + Meta‑RL)

**구조**: MARL + 메타‑러닝 기반의 훈련-미세조정 구조&#x20;

1. **Temporal Reward 설계**

   * 연속적 우선순위(task precedence)에 맞춘 시간 기반 보상
2. **중추 에이전트 학습**

   * MADDPG를 통해 협업 MDP 해결 방식으로 정책 학습
3. **Meta‑Learning (Reptile)**

   * 여러 태스크 분포로 초기화 → 새로운 task에 빠르게 적응(ft tuning)

> 요약: 플래너 구조는 플래닝 알고리즘이 아닌 **학습 체계**: 시간 기반 MARL + 메타‑러닝을 통한 **다양한 시나리오 적응성** 강화입니다.

---

## 5. AdaPlanner (AdaPlanner: Adaptive Planning from Feedback with LMs)

**구조**: 자기‑피드백 루프 기반 LLM 플래너 ([mdpi.com][6])

1. **Plan 생성**: LLM이 브레인스토밍 방식으로 초기 계획 작성
2. **피드백 수집**: 성공/실패 케이스를 통해 내부 피드백 받아 Plan refinement
3. **Skill Discovery**: 성공 사례를 few-shot 예시로 활용해 효율성 증대
4. **in‑plan / out‑of‑plan 수정**: 계획 내부 단계 고도화 또는 완전 재작성

> 요약: AdaPlanner는 **plan → 환경 feedback → in/out 점진 수정**의 반복을 통해 적응성을 보장합니다.

---

### 🔍 비교 요약

| 논문            | 플래너 구조                                                             |
| ------------- | ------------------------------------------------------------------ |
| MACI          | Meta‑Planner → 역할 에이전트 배치 → 실행 모니터링                                |
| ALAS          | 역할 정의 → 실행 에이전트 + 보상 프로필 → Validator agent                         |
| PlanGEN       | Constraint Extractor → Verifier → Algorithm Selector → Replan loop |
| ChronoPlanner | MADDPG 기반 MARL + Meta‑RL 초기화 구조                                    |
| AdaPlanner    | LLM Plan → Feedback 수집 → In/Out 단계 수정 루프                           |

---

더 깊게 보길 원하는 논문이 있다면 말씀해 주세요. 예:

* MACI의 메타플래너 내부 구현 details
* ALAS 보상과 validation 로직의 formal spec
* ChronoPlanner reward 설계 수식
  등을 추가로 찾아 드릴 수 있어요!

[1]: https://arxiv.org/abs/2305.16653?utm_source=chatgpt.com "AdaPlanner: Adaptive Planning from Feedback with Language Models"
[2]: https://www.themoonlight.io/ko/review/maci-multi-agent-collaborative-intelligence-for-adaptive-reasoning-and-temporal-planning?utm_source=chatgpt.com "[논문 리뷰] MACI: Multi-Agent Collaborative Intelligence for Adaptive Reasoning and Temporal Planning"
[3]: https://www.themoonlight.io/en/review/maci-multi-agent-collaborative-intelligence-for-adaptive-reasoning-and-temporal-planning?utm_source=chatgpt.com "[Literature Review] MACI: Multi-Agent Collaborative Intelligence for Adaptive Reasoning and Temporal Planning"
[4]: https://arxiv.org/abs/2505.12501?utm_source=chatgpt.com "ALAS: A Stateful Multi-LLM Agent Framework for Disruption-Aware Planning"
[5]: https://www.researchgate.net/publication/391851064_ALAS_A_Stateful_Multi-LLM_Agent_Framework_for_Disruption-Aware_Planning?utm_source=chatgpt.com "(PDF) ALAS: A Stateful Multi-LLM Agent Framework for Disruption-Aware Planning"
[6]: https://www.mdpi.com/2076-3417/13/16/9174?utm_source=chatgpt.com "Multi-Agent Chronological Planning with Model-Agnostic Meta Reinforcement Learning"



좋은 질문입니다!
\*\*"Multi-Agent 시스템에서 planner가 각 agent마다 하나씩 있는지, 아니면 전체를 하나의 planner가 관리하는지?"\*\*는 시스템 설계 방식에 따라 달라집니다. 아래에 **대표적인 두 가지 구조**와 최신 논문에서 어떤 방식을 채택했는지를 정리해드릴게요.

---

## ✅ 두 가지 대표적인 Multi-Agent Planning 구조

| 구조 유형                          | 설명                                             | 장단점                                     |
| ------------------------------ | ---------------------------------------------- | --------------------------------------- |
| **중앙집중형(Centralized Planner)** | 하나의 중앙 planner가 모든 agent의 계획을 생성·조율함           | ✅ 전체 최적화 용이<br>❌ 유연성/확장성 낮음, 실패 시 전체 중단 |
| **분산형(Decentralized Planner)** | 각 agent가 자체 planner를 갖고 독립적으로 또는 국지적으로 협의하며 계획 | ✅ 유연성/복원력 높음<br>❌ 전체 최적화 어렵고 충돌 가능성 있음  |

---

## 🧠 주요 논문 구조 정리

| 논문                               | 구조                 | 설명                                                                      |
| -------------------------------- | ------------------ | ----------------------------------------------------------------------- |
| **MACI** (2024)                  | **부분 중앙집중형**       | 메타플래너가 전체 플랜을 구성한 후 → 역할별로 agent에 task 분배. 각 agent는 자체적으로 실행·검증 로직을 가짐  |
| **ALAS** (2024)                  | **완전 분산형**         | 모든 agent가 독립적인 planner + 상태 추적기 + 보상 평가기를 갖춤. Coordination 없이도 로컬 보정 가능 |
| **PlanGEN**                      | **중앙집중형**          | 제약 추출, 계획 생성, 검증, 재계획 모두 하나의 LLM planner가 수행 (단일 주체)                    |
| **ChronoPlanner (Meta-RL 기반)**   | **학습은 중앙, 실행은 분산** | 학습(초기화)은 중앙에서 진행되지만, 실행 시에는 각 agent가 독립된 정책을 사용                         |
| **AdaPlanner** (LLM Feedback 기반) | **단일 planner 중심**  | 하나의 LLM이 반복적으로 feedback을 수용하여 계획을 개선함 (multi-agent는 아님)                 |

---

## 🔄 요약

| 상황                         | 사용 구조                                                         |
| -------------------------- | ------------------------------------------------------------- |
| **작은 규모 + 정적 환경**          | 중앙집중형 planner (PlanGEN 등)                                     |
| **크고 복잡한 환경, 동적 task**     | 분산형 or 하이브리드 구조 (MACI, ALAS 등)                                |
| **LLM 기반 agent framework** | 대부분 **분산형** → 각 agent가 LLM 호출 및 자체 판단 수행 (예: ALAS, AutoGen 등) |

---

## 💡 결론

> **"모든 agent마다 planner가 있는가?"**
> ▶ **최신 Multi-Agent 시스템에서는 '역할 또는 상황에 따라 planner를 분산'시키는 하이브리드 구조가 많습니다.**

* 작은 시스템 → **하나의 중앙 planner**
* 복잡한 시스템 (특히 LLM, 강화학습 기반) → **agent마다 planner 내장**
* 실시간 실패 대응, 적응성 요구 ↑ → **분산형 선호**

---

필요하시다면, "agent 간 planner 협의 프로토콜", "LLM 기반 agent 협업 시 planner-실행자 분리 여부", "분산형 planner 간 충돌 해결 방식" 같은 구체 기술도 정리해드릴 수 있습니다.


당신이 구상하는 구조는 다음과 같이 정리할 수 있습니다:

---

## 🌳 **중앙 Planner + 하위 Agent별 Planner 구조 = 계층적 Tree of Thought형 Multi-Agent Planning**

### ✅ 개념 요약

* **중앙 Planner**: 전체 목표를 나누고 Subtask를 설계
* **Agent별 Planner**: 받은 Subtask를 스스로 계획하고 실행 (필요시 하위 Sub-planning도 가능)
* **협의/피드백**: 하위 계획 결과를 상위에 피드백하거나 수정 요청 가능

이건 일종의 **"Top-down reasoning with local autonomy"**, 혹은 **계층적 Tree-of-Thought 플래닝 구조**입니다.

---

## 📚 이 구조와 유사한 논문은?

| 논문                                     | 유사도               | 어떻게 유사한가?                                                                                           |
| -------------------------------------- | ----------------- | --------------------------------------------------------------------------------------------------- |
| **MACI (2024)**                        | ⭐⭐⭐⭐☆ (매우 유사)     | **중앙 메타플래너**가 역할/순서 정의 → **에이전트별 도메인 특화 Planner**가 실행 & 재계획 수행. 실패 시 로컬에서 자체 조정 or 모니터가 상위에 피드백     |
| **ALAS (2024)**                        | ⭐⭐⭐☆☆ (유사)        | 중앙 planner는 없음. 각 agent가 상태 추적기 + validator + role spec을 통해 **부분적 Tree-of-Thought 구조**를 재귀적으로 생성 가능 |
| **Tree of Thoughts (ToT, Yao et al.)** | ⭐⭐⭐⭐☆ (이론적 근거 제공) | ToT는 명시적으로 중앙 LLM이 하위 subthought를 구성하고 각 branch를 평가/확장함.                                            |
| **AutoGen / ChatDev 구조**               | ⭐⭐⭐☆☆             | 중앙 컨트롤러 + 역할별 LLM agent (코더, QA 등) → 일부는 재귀적으로 하위 계획 생성 가능                                          |

> 가장 유사한 구조는 **MACI**입니다. MACI는 중앙에서 “역할 기반 토대”를 만들고, 각 역할이 **실행자이자 부분 플래너**로 동작합니다.

---

## 🆚 장단점 비교: 중앙 + 로컬 플래너 구조 (Tree of Thought형)

| 구분                              | 장점                                                                                                          | 단점                                                                                               |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **계층적 플래너 구조**<br>(중앙 + 로컬 플래너) | ✅ 전체 최적화 + 로컬 유연성의 균형<br>✅ 실패 복원력 우수 (부분 실패 시 국소 수정 가능)<br>✅ Tree 형태로 분기한 reasoning 가능<br>✅ 재귀적 계획/검증 구현 가능 | ❌ 설계 복잡도 매우 높음<br>❌ 플래너 간 상태 동기화 문제 (context drift)<br>❌ 성능 측정 및 튜닝 난이도 상승<br>❌ 비결정적 계획 흐름 발생 위험 |
| **중앙집중형만 있음**                   | ✅ 전체 최적화 쉬움<br>✅ 시스템 관리/디버깅 단순                                                                              | ❌ 적응성 떨어짐<br>❌ 분산 상황 대응에 취약                                                                      |
| **완전 분산형만 있음**                  | ✅ 확장성 높음<br>✅ 유연성, 장애 복원력 우수                                                                                | ❌ 전체 계획 정합성 유지 어려움<br>❌ 통합 목표 최적화 어려움                                                            |

---

## 🧠 추천 아키텍처 설계 방향 (구상 기반)

```mermaid
graph TD
    A[Central Planner (Meta-Planner)] --> B1[Agent 1 Planner]
    A --> B2[Agent 2 Planner]
    B1 --> C1[Sub-sub plan (if needed)]
    B2 --> C2[Execution & Adaptation]
    B1 -- Feedback --> A
    B2 -- Result --> A
```

* **Central Planner (A)**: 목표 분할, 제약 조정, 에이전트 할당
* **Agent Planner (B)**: 할당받은 역할/task에 대한 세부 계획 생성 및 재계획
* **재귀성 (C)**: 필요시 sub-plan 구성 (즉, depth > 2 가능)
* **피드백 루프**: 결과/오류/실패가 상위로 피드백되어 전체 plan이 적응됨

---

## 🔧 구현 참고

| 구현 요소         | 도구 or 기법                                   |
| ------------- | ------------------------------------------ |
| 중앙 Planner    | LangGraph, LLM + Tree of Thought prompting |
| Agent Planner | LangChain agent, AutoGen role agent        |
| 실행 로직         | Tool 사용 (MCP, DB 접근 등)                     |
| 검증            | 독립 Validator agent or tool                 |
| 토큰 절약         | ReAct + scratchpad or 외부 vector memory 연동  |

---

## ✍️ 결론

> 당신이 생각한 구조는 **MACI**의 구조에 가장 유사하며, **Tree-of-Thought + Multi-Agent + Adaptive Planner**를 통합한 최신 추세에 부합합니다.
> 이러한 계층형 구조는 복잡한 문제를 인간처럼 “생각을 나누고 평가하면서” 해결하는 방식에 가깝기 때문에 앞으로도 많이 채택될 전망입니다.

필요하면 다음도 도와드릴 수 있어요:

* MACI 구조 기반 구현 예시
* Tree of Thought 프롬프트 설계 템플릿
* Central + Local Planner 통신 구조 설계 (예: JSON 스키마)
* LangGraph로 구성된 예제 워크플로우

언제든지 말씀해주세요!
