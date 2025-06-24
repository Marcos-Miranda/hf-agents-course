# ruff: noqa: E402
from dotenv import load_dotenv

load_dotenv()

import argparse

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.chat_models import ChatOllama
from langfuse.langchain import CallbackHandler  # type: ignore

from app.agent import Answer, CompiledStateGraph, create_agent
from app.prompts import AGENT_SYSTEM_MESSAGE

langfuse_handler = CallbackHandler()


def answer_question(agent: CompiledStateGraph, question_data: dict, question_idx: int) -> Answer | None:
    question = question_data["question"]
    if question_data["file_name"]:
        question += f"\nFile path: files/{question_data['file_name']}"
    for event in agent.stream(
        {"messages": [SystemMessage(content=AGENT_SYSTEM_MESSAGE), HumanMessage(content=question)]},
        stream_mode="values",
        config={
            "callbacks": [langfuse_handler],
            "metadata": {
                "langfuse_tags": [question_data["task_id"], f"question-{question_idx}"],
            },
        },
    ):
        event["messages"][-1].pretty_print()
    return event["structured_output"]


def main(ollama_model: str, temperature: float) -> None:
    questions = httpx.get("https://agents-course-unit4-scoring.hf.space/questions").json()
    chat_model = ChatOllama(model=ollama_model, temperature=temperature)
    agent = create_agent(chat_model)
    for idx, question_data in enumerate(questions):
        try:
            answer = answer_question(agent, question_data, idx)
            assert answer is not None, "Could not get the structured output from the agent."
        except Exception as e:
            print(f"Error processing question {question_data['id']}: {e}")
            continue
        print(f"\nQuestion: {question_data['question']}")
        print(f"Answer: {answer.final_answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama_model", type=str, default="qwen3:14b")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    main(ollama_model=args.ollama_model, temperature=args.temperature)
