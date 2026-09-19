from pathlib import Path

from app.ingestion import load_corpus
from app.query import build_retrieval_query
from app.schemas import ChatTurn


def main() -> None:
    corpus = load_corpus(Path("data"))
    assert len(corpus) == 40
    assert sum(x.source_type == "faq" for x in corpus) == 15
    assert sum(x.source_type == "policy" for x in corpus) == 10
    assert sum(x.source_type == "ticket" for x in corpus) == 15

    history = [
        ChatTurn(role="user", content="Can I get a refund for an individual course?"),
        ChatTurn(role="assistant", content="The standard window is generally 14 days."),
    ]
    rewritten = build_retrieval_query("What if I bought mine 20 days ago?", history)
    assert "Can I get a refund" in rewritten
    print("Smoke check passed: corpus=40 and multi-turn rewrite works.")


if __name__ == "__main__":
    main()
