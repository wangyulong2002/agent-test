from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage
from datetime import datetime
import os
from langchain.agents import AgentState
from typing import NotRequired
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

# ============ 模型配置（统一使用 agnes-2.0-flash） ============
# API Key / 接入地址请在 .env 中填写：AGNES_API_KEY、AGNES_BASE_URL（OpenAI 兼容端点）
AGNES_MODEL = os.getenv("AGNES_MODEL", "agnes-2.0-flash")
AGNES_API_KEY = os.getenv("AGNES_API_KEY", "")    # 用户自行填写
AGNES_BASE_URL = os.getenv("AGNES_BASE_URL", "")  # 用户自行填写（OpenAI 兼容端点）

# 定义自定义State结构
class CustomState(AgentState):
    """Agent的任务状态"""
    model_call_count: NotRequired[int]  # 模型调用次数
    session_start: NotRequired[str]  # 会话开始时间
    
@tool
def update_state(runtime: ToolRuntime):
    """A tool that update agent state"""
    # 获取state中的历史消息
    messages = runtime.state['messages']
    # 消息数量
    message_count = len(messages)
    # 组织结果
    command = {
        "model_call_count": runtime.state.get("model_call_count", 0) + 1,
        "messages": [ToolMessage("Successfully updated agent state", tool_call_id=runtime.tool_call_id)]
    }
    # 判断是否是第一次
    if message_count <= 2:
        command['session_start'] = datetime.now()

    return Command(update=command)
    
agent = create_agent(
    model=init_chat_model(
        model=AGNES_MODEL,
        model_provider="openai",            # agnes 走 OpenAI 兼容接口
        api_key=AGNES_API_KEY,
        base_url=AGNES_BASE_URL or None,
    ),
    tools=[update_state],
    state_schema=CustomState,
    checkpointer=InMemorySaver(),
    system_prompt="你是一个热心的助手，你需要在每次请求时调用update_state工具以更新任务状态。"
)

config = {"configurable": {"thread_id": "1"}}
response = agent.invoke(
    {"messages": [HumanMessage(content="Hi, my name is 虎哥")]},
    config
)

for message in response['messages']:
    message.pretty_print()