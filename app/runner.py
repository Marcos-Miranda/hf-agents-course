# ruff: noqa: E402
from dotenv import load_dotenv

load_dotenv()

import argparse
import os
import pprint
import time

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_ollama.chat_models import ChatOllama
from langfuse.langchain import CallbackHandler  # type: ignore

from app.agent import Answer, CompiledStateGraph, create_agent
from app.prompts import AGENT_SYSTEM_MESSAGE

langfuse_handler = CallbackHandler()
COURSE_API_URL = "https://agents-course-unit4-scoring.hf.space"


def answer_question(
    agent: CompiledStateGraph, question_data: dict, question_idx: int, submission: bool, agent_provider: str
) -> Answer | None:
    question = question_data["question"]
    if question_data["file_name"]:
        question += f"\nFile path: files/{question_data['file_name']}"
    tags = [question_data["task_id"], f"question-{question_idx}", agent_provider]
    if submission:
        tags.append("submission")
    return agent.invoke(
        {"messages": [SystemMessage(content=AGENT_SYSTEM_MESSAGE), HumanMessage(content=question)]},
        config={"callbacks": [langfuse_handler], "metadata": {"langfuse_tags": tags}},
    )["structured_output"]


def submit_answers(answers: list[dict[str, str]]) -> dict:
    data = {
        "username": os.environ["USERNAME"],
        "agent_code": os.environ["AGENT_CODE"],
        "answers": answers,
    }
    resp = httpx.post(f"{COURSE_API_URL}/submit", json=data)
    resp.raise_for_status()
    return resp.json()


def main(
    agent_provider: str,
    ollama_model_name: str,
    google_model_name: str,
    temperature: float,
    only_index: int,
    submit: bool,
) -> None:
    questions = httpx.get(f"{COURSE_API_URL}/questions").json()
    if only_index >= 0:
        questions = [questions[only_index]]
    ollama_model = ChatOllama(model=ollama_model_name, temperature=temperature, extract_reasoning=True)
    google_model = ChatGoogleGenerativeAI(model=google_model_name, temperature=temperature)  # , include_thoughts=True)
    agent = create_agent(
        agent_model=ollama_model if agent_provider == "ollama" else google_model, aux_model=ollama_model
    )
    answers = []
    for idx, question_data in enumerate(questions):
        try:
            answer = answer_question(agent, question_data, idx, submit, agent_provider)
            assert answer is not None, "Could not get the structured output from the agent."
        except Exception as e:
            print(f"Error processing question {question_data['task_id']}: {e}")
            continue
        print(f"\nQuestion: {question_data['question']}")
        print(f"Answer: {answer.final_answer}")
        answers.append(
            {
                "task_id": question_data["task_id"],
                "submitted_answer": answer.final_answer.strip(),
            }
        )
        if agent_provider == "google" and idx != len(questions) - 1:
            print("Waiting for 60 seconds to avoid rate limiting...")
            time.sleep(60)
    if submit:
        results = submit_answers(answers)
        print("\nSubmission results:")
        pprint.pprint(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-provider", type=str, default="ollama", choices=["ollama", "google"])
    parser.add_argument("--ollama-model", type=str, default="qwen3:14b")
    parser.add_argument("--google-model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--only-index", type=int, default=-1)
    parser.add_argument("--submit", action="store_true", default=False)
    args = parser.parse_args()
    main(
        agent_provider=args.agent_provider,
        ollama_model_name=args.ollama_model,
        google_model_name=args.google_model,
        temperature=args.temperature,
        only_index=args.only_index,
        submit=args.submit,
    )
