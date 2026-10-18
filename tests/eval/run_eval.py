"""End-to-end evaluation in two stages: the agent answers every case, then an LLM judge grades the answers.

Needs the real .env (Anthropic key; Snowflake for `generate`). Conversation state is kept in memory, no Postgres.

    generate   run the real agent on the cases and append each answer to a JSONL file as soon as it exists
    judge      grade the answers from such a file; can be repeated with another judge prompt or model
    run        generate, then judge

    uv run python tests/eval/run_eval.py generate --only A1 B1
    uv run python tests/eval/run_eval.py judge tests/eval/results/answers-20260101-120000.jsonl
    uv run python tests/eval/run_eval.py run --category rankings
"""

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from us_population_agent.agent.graph import build_graph
from us_population_agent.config import get_settings
from us_population_agent.llm.factory import build_llm
from us_population_agent.snowflake_client.columns import ColumnCatalog
from us_population_agent.snowflake_client.connection import open_connection
from us_population_agent.snowflake_client.query import QueryRunner
from us_population_agent.snowflake_client.search import CatalogSearch

EVAL_DIR = Path(__file__).parent
CASES_FILE = EVAL_DIR / "cases.yaml"
RESULTS_DIR = EVAL_DIR / "results"

JUDGE_PROMPT = """\
You are grading an assistant that answers questions about US Census (ACS 2020 5-year) data.
Compare the assistant's answer with the reference answer and decide whether the assistant's answer is correct.

GENERAL RULES
- Judge substance, not wording. Formatting and rounding differences are fine (1.25 million = 1,250,884 only if the
  assistant's number is clearly the same quantity; 1,250,884 vs 1,250,000 is not an exact match, see below).
- Extra correct context does not make an answer wrong. A number that contradicts the reference does.
- If the assistant says it could not find or compute the data, the answer is incorrect, unless the reference
  expects exactly that.

CATEGORY RULES
{category_rules}

Think first and write your reasoning, then give the verdict.
"""

CATEGORY_RULES = {
    "counts": "Counts must match the reference exactly.",
    "rates": (
        "Percentages must match the reference within 0.01 percentage points. Counts must match exactly. If the "
        "reference gives two denominators, either is acceptable as long as the answer says which one it uses and "
        "its figure matches that one."
    ),
    "age": "Counts must match exactly, percentages within 0.01 percentage points.",
    "rankings": "Every listed item and its order must match the reference. Values within 0.01 percentage points.",
    "medians": (
        "Do not require exact numbers. The answer is correct if the ordering of the compared values matches the "
        "reference AND the answer says the figures are approximate or estimated."
    ),
    "conversation": "Percentages must match the reference within 0.01 percentage points.",
    "clarification": (
        "The assistant must ask a clarifying question or explain what is missing, along the lines of the "
        "reference. It must not state any statistical figure."
    ),
    "meta": "The assistant must explain it as in the reference. It must not run new numbers or state new figures.",
    "off_topic": "The assistant must decline or explain the limitation in the reference. It must not state a figure.",
}


class Verdict(BaseModel):
    # `reasoning` comes first so the judge explains itself before committing to a verdict.
    reasoning: str = Field(description="Short comparison of the assistant's answer with the reference.")
    correct: bool


def stamp() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- Stage 1: generate -----------------------------------------------------------------------------------------


def load_cases(path: Path, only: list[str] | None, categories: list[str] | None) -> list[dict[str, Any]]:
    cases = yaml.safe_load(path.read_text())
    if only:
        cases = [c for c in cases if c["id"] in only]
    if categories:
        cases = [c for c in cases if c["category"] in categories]
    return cases


def ask_agent(graph, case: dict[str, Any]) -> str:
    """Replay the earlier turns of the conversation, then ask the question; return the last reply."""
    config = {"configurable": {"thread_id": f"eval-{case['id']}-{uuid.uuid4().hex[:8]}"}}
    # A history entry is either a question to replay through the agent, or a fixed {role, content} message.
    seeded = []
    for turn in case.get("history", []):
        if isinstance(turn, str):
            graph.invoke({"messages": [HumanMessage(turn)]}, config)
        else:
            message_class = AIMessage if turn["role"] == "assistant" else HumanMessage
            seeded.append(message_class(turn["content"]))
    result = graph.invoke({"messages": [*seeded, HumanMessage(case["question"])]}, config)
    return str(result["messages"][-1].content)


def generate(args: argparse.Namespace) -> Path | None:
    """Ask the agent every selected case. Each answer is appended to the file right away, so a crash loses nothing."""
    cases = load_cases(args.cases, args.only, args.category)
    if not cases:
        print("No cases selected.", file=sys.stderr)
        return None

    output = args.output or RESULTS_DIR / f"answers-{stamp()}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_jsonl(output) if r.get("answer") is not None}  # lets --output resume a run
    todo = [c for c in cases if c["id"] not in done]
    print(f"{len(todo)} to run, {len(cases) - len(todo)} already answered in {output}")

    settings = get_settings()
    llm = build_llm(effort="low")
    sql_llm = build_llm(settings.anthropic_strong_model_name)
    with open_connection(settings) as connection:
        graph = build_graph(
            llm,
            CatalogSearch(connection, settings),
            ColumnCatalog(connection, settings),
            QueryRunner(connection, settings),
            sql_llm,
            settings,
            InMemorySaver(),
        )
        for i, case in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {case['id']}: {case['question']}", flush=True)
            record: dict[str, Any] = {k: case[k] for k in ("id", "category", "question", "reference")}
            try:
                record["answer"] = ask_agent(graph, case)
            except Exception as exc:  # noqa: BLE001  # one failing case must not stop the run
                record.update(answer=None, error=f"{type(exc).__name__}: {exc}")
                print(f"    ERROR: {record['error']}", flush=True)
            with output.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Answers saved to {output}")
    return output


# --- Stage 2: judge --------------------------------------------------------------------------------------------


def judge_answer(judge_llm: BaseChatModel, record: dict[str, Any]) -> Verdict:
    rules = CATEGORY_RULES.get(record["category"], "Use the general rules only.")
    evidence = (
        f"QUESTION:\n{record['question']}\n\n"
        f"REFERENCE ANSWER:\n{record['reference']}\n\n"
        f"ASSISTANT'S ANSWER:\n{record['answer']}"
    )
    messages = [("system", JUDGE_PROMPT.format(category_rules=rules)), ("human", evidence)]
    return judge_llm.with_structured_output(Verdict, method="json_schema").invoke(messages)  # ty: ignore[invalid-return-type]


def judge(args: argparse.Namespace, answers_file: Path) -> bool:
    """Grade every answer in the file. Returns True when all cases are correct."""
    records = list({r["id"]: r for r in read_jsonl(answers_file)}.values())  # last record per id wins
    if not records:
        print(f"No answers in {answers_file}.", file=sys.stderr)
        return False

    judge_llm = build_llm(args.judge_model or get_settings().anthropic_strong_model_name)
    results = []
    for i, record in enumerate(records, 1):
        print(f"[{i}/{len(records)}] {record['id']}", flush=True)
        result = dict(record)
        if record.get("answer") is None:  # the agent crashed; nothing to judge
            result.update(correct=False, reasoning=f"Agent error: {record.get('error')}")
        else:
            try:
                verdict = judge_answer(judge_llm, record)
                result.update(correct=verdict.correct, reasoning=verdict.reasoning)
            except Exception as exc:  # noqa: BLE001  # one failing case must not stop the run
                result.update(correct=False, reasoning=f"Judge error: {type(exc).__name__}: {exc}")
        print(f"    {'PASS' if result['correct'] else 'FAIL'}", flush=True)
        results.append(result)

    print_summary(results)
    output = args.output or answers_file.with_name(f"verdicts-{stamp()}.json")
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nVerdicts saved to {output}")
    return all(r["correct"] for r in results)


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\n=== Summary ===")
    for category in dict.fromkeys(r["category"] for r in results):
        group = [r for r in results if r["category"] == category]
        print(f"{category:<15} {sum(r['correct'] for r in group)}/{len(group)}")
    print(f"{'TOTAL':<15} {sum(r['correct'] for r in results)}/{len(results)}")

    for r in (r for r in results if not r["correct"]):
        print(f"\n--- FAIL {r['id']}: {r['question']}")
        print(f"reference: {r['reference']}")
        print(f"answer:    {r.get('answer')}")
        print(f"judge:     {r['reasoning']}")


# --- CLI -------------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    selection = argparse.ArgumentParser(add_help=False)
    selection.add_argument("--cases", type=Path, default=CASES_FILE, help="cases file (default: cases.yaml)")
    selection.add_argument("--only", nargs="+", metavar="ID", help="run only these case ids")
    selection.add_argument("--category", nargs="+", metavar="NAME", help="run only these categories")
    judging = argparse.ArgumentParser(add_help=False)
    judging.add_argument("--judge-model", help="judge model (default: ANTHROPIC_STRONG_MODEL_NAME)")

    gen = commands.add_parser("generate", parents=[selection], help="stage 1: collect the agent's answers")
    gen.add_argument("--output", type=Path, help="answers JSONL; an existing file is resumed (default: new file)")

    jdg = commands.add_parser("judge", parents=[judging], help="stage 2: grade a file of answers")
    jdg.add_argument("answers", type=Path, help="JSONL file produced by `generate`")
    jdg.add_argument("--output", type=Path, help="verdicts JSON (default: next to the answers file)")

    both = commands.add_parser("run", parents=[selection, judging], help="generate, then judge")
    both.add_argument("--output", type=Path, help="answers JSONL (default: new file)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "judge":
        return 0 if judge(args, args.answers) else 1

    answers_file = generate(args)
    if answers_file is None:
        return 2
    if args.command == "generate":
        return 0

    args.output = None  # `--output` named the answers file; the verdicts go next to it
    return 0 if judge(args, answers_file) else 1


if __name__ == "__main__":
    raise SystemExit(main())
