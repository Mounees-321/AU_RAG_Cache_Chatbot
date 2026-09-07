"""
evaluate.py — Phase 8: Evaluation Harness

Runs eval_qa_set.py's questions through the REAL pipeline (retrieval,
re-ranking, generation, routing) and scores three things:

    1. Keyword accuracy   — did the generated answer contain the expected
                             facts? (proxy for "was retrieval + generation
                             actually correct", not just "did it answer")
    2. Routing accuracy   — did the agent correctly decide KB vs. search?
    3. Response time       — how long each query took end-to-end

This is what turns your resume's "95%+ accuracy" claim from an aspiration
into a number backed by a real, re-runnable measurement — and gives you
something honest to say if an interviewer asks "how did you measure that?"

Usage:
    python evaluate.py
Output:
    Prints a per-question breakdown + summary stats to the console
    Saves detailed results to eval_results.json
"""

import json
import logging
import time

import router
from eval_qa_set import EVAL_SET
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def keyword_match(answer: str, expected_keywords: list[str]) -> tuple[bool, list[str]]:
    """
    Check whether all expected keywords appear in the answer (case-insensitive).
    Returns (all_matched, list_of_missing_keywords).
    """
    if not expected_keywords:
        return True, []  # nothing specific to check (e.g. search-fallback questions)

    answer_lower = answer.lower()
    missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
    return len(missing) == 0, missing


def evaluate_single(item: dict) -> dict:
    """Run one eval question through the real pipeline and score it."""
    question = item["question"]
    expected_keywords = item["expected_keywords"]
    expected_search = item["should_use_search"]

    logger.info(f"Evaluating: {question!r}")
    start = time.time()

    try:
        result = router.route_and_answer(question)
    except Exception as e:
        logger.error(f"Evaluation failed for {question!r}: {e}")
        return {
            "question": question,
            "error": str(e),
            "keyword_match": False,
            "routing_correct": False,
            "response_time": time.time() - start,
        }

    elapsed = time.time() - start

    matched, missing = keyword_match(result["answer"], expected_keywords)
    routing_correct = result["used_search"] == expected_search

    return {
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
        "used_search": result["used_search"],
        "expected_search": expected_search,
        "routing_correct": routing_correct,
        "keyword_match": matched,
        "missing_keywords": missing,
        "response_time": round(elapsed, 2),
    }


def run_evaluation() -> dict:
    """Run the full eval set and compute summary statistics."""
    logger.info(f"Running evaluation on {len(EVAL_SET)} questions...")

    results = [evaluate_single(item) for item in EVAL_SET]

    total = len(results)
    keyword_correct = sum(1 for r in results if r.get("keyword_match"))
    routing_correct = sum(1 for r in results if r.get("routing_correct"))
    errors = sum(1 for r in results if "error" in r)
    avg_time = sum(r.get("response_time", 0) for r in results) / total if total else 0

    summary = {
        "total_questions": total,
        "keyword_accuracy": round(keyword_correct / total * 100, 1) if total else 0,
        "routing_accuracy": round(routing_correct / total * 100, 1) if total else 0,
        "errors": errors,
        "avg_response_time_seconds": round(avg_time, 2),
    }

    return {"summary": summary, "results": results}


def print_report(eval_output: dict) -> None:
    summary = eval_output["summary"]
    results = eval_output["results"]

    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    for r in results:
        status = "✓" if r.get("keyword_match") and r.get("routing_correct") else "✗"
        print(f"\n[{status}] {r['question']}")
        if "error" in r:
            print(f"    ERROR: {r['error']}")
            continue
        print(f"    Used search: {r['used_search']} (expected: {r['expected_search']}) "
              f"{'✓' if r['routing_correct'] else '✗ MISMATCH'}")
        if r.get("missing_keywords"):
            print(f"    Missing expected keywords: {r['missing_keywords']}")
        print(f"    Response time: {r['response_time']}s")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Questions evaluated:     {summary['total_questions']}")
    print(f"  Keyword accuracy:        {summary['keyword_accuracy']}%")
    print(f"  Routing accuracy:        {summary['routing_accuracy']}%")
    print(f"  Errors:                  {summary['errors']}")
    print(f"  Avg response time:       {summary['avg_response_time_seconds']}s")
    print("=" * 70 + "\n")

    print("NOTE: 'Keyword accuracy' is a proxy metric, not the same as your")
    print("earlier vector-retrieval-only metric. Report this number, or run")
    print("a stricter human-graded eval, before using any accuracy figure")
    print("on your resume. See docstring in this file for methodology notes.\n")


def main():
    eval_output = run_evaluation()
    print_report(eval_output)

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_output, f, ensure_ascii=False, indent=2)
    logger.info("Saved detailed results -> eval_results.json")


if __name__ == "__main__":
    main()