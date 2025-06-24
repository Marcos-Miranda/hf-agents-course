from typing import Annotated, Literal

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from app.prompts import FINAL_ANSWER_FORMAT
from app.tools import (
    DescribeImageTool,
    get_website_content,
    get_wikipedia_page_content,
    python_repl_tool,
    transcribe_audio,
)


class Answer(BaseModel):
    final_answer: str = Field(description=FINAL_ANSWER_FORMAT)


class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    structured_output: Answer | None = None


def create_agent(chat_model: BaseChatModel) -> CompiledStateGraph:
    tools = [
        transcribe_audio,
        DescribeImageTool(),
        python_repl_tool,
        DuckDuckGoSearchResults(output_format="json"),
        get_wikipedia_page_content,
        get_website_content,
    ]
    tools_node = ToolNode(tools=tools)
    chat_model_with_tools = chat_model.bind_tools(tools)
    model_with_structured_output = chat_model.with_structured_output(Answer)

    def chat_model_node(state: State) -> dict:
        return {"messages": [chat_model_with_tools.invoke(state.messages)]}

    def tools_or_format_answer_condition(state: State) -> Literal["format_answer", "tools"]:
        try:
            ai_message = state.messages[-1]
        except IndexError:
            raise ValueError(f"No messages found in input state: {state}") from None
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return "format_answer"

    def format_answer_node(state: State) -> dict:
        messages = [state.messages[0], state.messages[-1]]
        return {"structured_output": model_with_structured_output.invoke(messages)}

    graph_builder = StateGraph(State)
    graph_builder.add_node("chat_model", chat_model_node)
    graph_builder.add_node("tools", tools_node)
    graph_builder.add_node("format_answer", format_answer_node)
    graph_builder.add_edge(START, "chat_model")
    graph_builder.add_conditional_edges("chat_model", tools_or_format_answer_condition)
    graph_builder.add_edge("tools", "chat_model")
    graph_builder.add_edge("format_answer", END)
    return graph_builder.compile()
