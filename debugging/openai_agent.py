from typing import Annotated
from typing_extensions import TypedDict

from langchain_groq import ChatGroq   # ✅ Use Groq instead of OpenAI
from langgraph.graph import END, START
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages

from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ✅ Set GROQ API Key from .env
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")  # Optional, for LangSmith if used

# ✅ Define shared state structure
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ✅ Instantiate Groq model (choose from `llama3-8b-8192`, `mixtral-8x7b-32768`, etc.)
model = ChatGroq(model="llama3-8b-8192", temperature=0)

# ---------- Basic Graph ----------
def make_default_graph():
    graph_workflow = StateGraph(State)

    def call_model(state):
        return {"messages": [model.invoke(state["messages"])]}

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_edge("agent", END)
    graph_workflow.add_edge(START, "agent")

    return graph_workflow.compile()

# ---------- Tool-Calling Graph ----------
def make_alternative_graph():
    @tool
    def add(a: float, b: float):
        """Adds two numbers."""
        return a + b

    tool_node = ToolNode([add])
    model_with_tools = model.bind_tools([add])

    def call_model(state):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def should_continue(state: State):
        if state["messages"][-1].tool_calls:
            return "tools"
        return END

    graph_workflow = StateGraph(State)
    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tools", tool_node)
    graph_workflow.add_edge("tools", "agent")
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_conditional_edges("agent", should_continue)

    return graph_workflow.compile()

# ✅ Run the alternative graph
agent = make_alternative_graph()
