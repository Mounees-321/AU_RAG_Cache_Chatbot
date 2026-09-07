"""
eval_qa_set.py — Phase 8: Evaluation Dataset

A small labeled set of question -> expected-answer-signal pairs, used by
evaluate.py to measure real retrieval and answer quality against YOUR
actual scraped data.

IMPORTANT: this file is a STARTING TEMPLATE. The questions/expected
keywords below are examples based on typical department-site content —
you MUST review and edit them to match what's actually on the real pages
you scraped, or the eval numbers will be meaningless.

How to fill this in:
    1. Run your pipeline once, look at what pages you actually scraped
       (check data/raw/*.json)
    2. Write 15-20 questions a real student might ask about YOUR
       scraped content
    3. For each, note 1-3 keywords/facts that MUST appear in a correct
       answer (e.g. a specific number, name, or fact from the source page)
    4. Also include a few questions you know are NOT answerable from your
       scraped pages (e.g. "What's the canteen menu today?") — these
       should trigger the search fallback, and are just as important to
       test as the answerable ones.
"""

EVAL_SET = [
    # ---- Answerable from the knowledge base (adjust to your real scraped content) ----
    {
        "question": "How many undergraduate students are in the IT department?",
        "expected_keywords": ["887"],
        "should_use_search": False,
    },
    {
        "question": "How many faculty members does the IT department have?",
        "expected_keywords": ["16"],
        "should_use_search": False,
    },
    {
        "question": "What are the thrust areas of research in the IT department?",
        "expected_keywords": ["Artificial Intelligence", "Machine Learning", "Cyber security"],
        "should_use_search": False,
    },
    {
        "question": "Who is the Head of the IT department?",
        "expected_keywords": ["Radha Senthilkumar"],
        "should_use_search": False,
    },
    {
        "question": "What is the contact email for the IT department?",
        "expected_keywords": ["hodit@annauniv.edu"],
        "should_use_search": False,
    },
    # TODO: add 10-15 more questions here based on your actual scraped pages,
    # covering: courses offered, academic schedule, teaching staff, facilities,
    # student association info, etc.

    # ---- NOT answerable from the KB — should trigger search fallback ----
    {
        "question": "What is the weather in Chennai today?",
        "expected_keywords": [],  # can't know this in advance — just check should_use_search
        "should_use_search": True,
    },
    {
        "question": "What is the latest news about Anna University admissions 2026?",
        "expected_keywords": [],
        "should_use_search": True,
    },
]