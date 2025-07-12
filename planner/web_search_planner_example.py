from dotenv import load_dotenv
import os
import asyncio
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langchain import hub
from langchain_openai import ChatOpenAI
import time
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool # <-- 추가
from langgraph.checkpoint.memory import MemorySaver
import operator
from typing import Annotated, List, Tuple
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import Literal
from langgraph.graph import END
from langgraph.graph import StateGraph, START

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

_tavily_search = TavilySearch(max_results=3)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
)

@tool
def tavily_search(query: str, **kwargs) -> str:
    """
    주어진 쿼리에 대해 Tavily 검색을 수행하고 결과를 반환합니다.
    검색 결과의 수를 조절하려면 k=숫자 인자를 사용할 수 있습니다.
    예: tavily_search(query="최신 AI 뉴스", k=5)
    """
    return _tavily_search.invoke(query, **kwargs)


tools = [tavily_search]
agent_executor = create_react_agent(llm, tools)

class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            """For the given objective, come up with a simple step by step plan. \
            This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
            The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
            하나의 step 에서는 반드시 하나의 작업만 해야하고, 여러가지 작업을 동시에 수행해서는 안돼.
            각 step 은 str 값이어야 하고, steps 라는 값에 List[str] 형태로 반환해야해.
            정보 검색이 필요하면 검색 step을 넣어줘.
            
            "user input : {user_input}"
            """,
        ),
    ]
)
planner = planner_prompt | llm.with_structured_output(Plan)


class Response(BaseModel):
    """Response to user."""
    response: str  = Field(
        description="Respond to user."
    )


replanner_prompt = ChatPromptTemplate.from_messages([(
    "user",
    """For the given objective, come up with a simple step by step plan. \
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
    하나의 step 에서는 반드시 하나의 작업만 해야하고, 여러가지 작업을 동시에 수행해서는 안돼.
    정보 검색이 필요하면 검색 step을 넣어줘.

    Your objective was this:
    {input}

    Your original plan was this:
    {plan}

    You have currently done the follow steps:
    {past_steps}

    Update your plan accordingly. If no more steps are needed and objective is finished, return empty list plan. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)])

replanner = replanner_prompt | llm.with_structured_output(Plan)

def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan: {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = agent_executor.invoke(
        {"messages": [("user", task_formatted)]}
    )
    time.sleep(10)

    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }


def plan_step(state: PlanExecute):
    plan_response = planner.invoke({"user_input": state["input"]})
    return {"plan": plan_response.steps}


def replan_step(state: PlanExecute):
    output = replanner.invoke(state)
    return {"plan": output.steps}


def make_response_step(state: PlanExecute):
    print(state["past_steps"])
    response_prompt = ChatPromptTemplate.from_messages([(
        "user",
        """
        User 질문 : {input}

        아래 과거 Step 들을 살펴보고, 위 User 질문에 대한 답변을 만들어서 response 에 리턴해줘.
        {past_steps}
        
        Generate a final report in markdown format. Write your response in Korean.
        """
    )])

    make_response_chain = response_prompt | llm.with_structured_output(Response)
    final_response = make_response_chain.invoke(state)
    print(final_response)
    return {"response" : final_response.response}


def should_end(state: PlanExecute):
    if not state["plan"]:
        return "make_response"
    else:
        return "agent"


workflow = StateGraph(PlanExecute)

workflow.add_node("planner", plan_step)
workflow.add_node("agent", execute_step)
workflow.add_node("replan", replan_step)
workflow.add_node("make_response", make_response_step)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "agent")
workflow.add_edge("agent", "replan")

workflow.add_edge("make_response", END)

workflow.add_conditional_edges(
    "replan",
    should_end,
    ["agent", "make_response"],
)

checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)


config = {"configurable": {"thread_id": "conversation-1"}, "recursion_limit": 50}
inputs = {"input": "2025년 7월 나스닥 100지수 향방에 대해서 여러가지 방면으로 검색해서 정밀 분석 진행해줘. 최근 주가, 주요 빅테크 PER, 이란 전쟁 상황, 유가 등을 포함해서."}

response = app.invoke(inputs, config)
print(response)


snapshot = app.get_state(config).values
print(snapshot["response"])

from IPython.display import Markdown

Markdown(snapshot["response"])



