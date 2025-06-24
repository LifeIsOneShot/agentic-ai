<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <title>LangGraph Plan-and-Execute 요약</title>
</head>
<body>
  <h1>LangGraph Plan-and-Execute 요약</h1>

  <h2>✅ 개요</h2>
  <p>LangGraph 써서 plan‑and‑execute 스타일 에이전트 만드는 법 다룸.<br>
  복잡한 작업은 장기 계획 → 단계별 실행 → 재계획 순서로 처리함.</p>

  <h2>🧠 핵심 개념</h2>
  <ul>
    <li><strong>Planner</strong>: 전체 작업 계획</li>
    <li><strong>Executor</strong>: 단계별 실행</li>
    <li><strong>Replanner</strong>: 계획 수정</li>
    <li><strong>Final Reporter</strong>: 보고서 생성</li>
  </ul>

  <h2>🛠 구성 요소</h2>
  <h3>도구 정의</h3>
  <pre><code>tools = [TavilySearch(max_results=3)]</code></pre>

  <h3>에이전트 생성</h3>
  <pre><code>agent_executor = create_react_agent(llm, tools, state_modifier=prompt)</code></pre>

  <h3>상태 정의</h3>
  <pre><code>class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: List[Tuple]
    response: str</code></pre>

  <h2>🔁 그래프 구성 예시</h2>
  <pre><code>workflow = StateGraph(PlanExecute)
workflow.add_node("planner", plan_step)
workflow.add_edge(START, "planner")</code></pre>

  <h2>📌 요약</h2>
  <ul>
    <li>Plan‑and‑Execute 방식으로 복잡한 태스크 처리</li>
    <li>LangGraph 사용 시 흐름 구조화 용이</li>
    <li>모델 자원도 단계별로 분리 가능</li>
  </ul>

  <p>🔗 <a href="https://wikidocs.net/270688" target="_blank">원문 보기</a></p>
</body>
</html>
