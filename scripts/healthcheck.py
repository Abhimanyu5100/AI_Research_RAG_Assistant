"""Verify the pieces the app depends on before you try to run it.

Every check here corresponds to a failure mode that is silent or confusing at
runtime: a decommissioned model, an empty index, BM25 parameters that do not
match the corpus, or missing `source` metadata that quietly disables citations.

    python scripts/healthcheck.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK, FAIL, WARN = "  \033[32mOK\033[0m  ", "  \033[31mFAIL\033[0m", "  \033[33mWARN\033[0m"
failures = 0


def report(status, label, detail=""):
    global failures
    if status is FAIL:
        failures += 1
    print(f"{status}  {label}" + (f" -- {detail}" if detail else ""))


def main() -> int:
    from src.config import (
        BM25_PATH, CHAT_MODEL, E5_USE_PREFIXES, GROQ_API_KEY, INDEX_NAME,
        PINECONE_API_KEY, SUMMARY_MODEL, TAVILY_API_KEY,
    )

    print("\n--- Configuration ---")
    for name, val in [("GROQ_API_KEY", GROQ_API_KEY), ("PINECONE_API_KEY", PINECONE_API_KEY)]:
        report(OK if val else FAIL, name, "" if val else "not set in .env")
    report(OK if TAVILY_API_KEY else WARN, "TAVILY_API_KEY",
           "" if TAVILY_API_KEY else "unset -- web search disabled")
    report(OK, "E5 prefixes", f"{'enabled' if E5_USE_PREFIXES else 'disabled'} "
                              "(must match how the index was built)")

    print("\n--- Groq models ---")
    try:
        from groq import Groq
        available = {m.id for m in Groq(api_key=GROQ_API_KEY).models.list().data}
        for label, model in [("chat", CHAT_MODEL), ("summary", SUMMARY_MODEL)]:
            report(OK if model in available else FAIL, f"{label} model {model}",
                   "" if model in available else "not available to this key")
    except Exception as exc:
        report(FAIL, "Groq API", str(exc)[:120])

    print("\n--- BM25 ---")
    report(OK if BM25_PATH.exists() else FAIL, "fitted parameters",
           str(BM25_PATH) if BM25_PATH.exists() else f"missing at {BM25_PATH}; run build_index.py")

    print("\n--- Pinecone ---")
    try:
        from src.agent.retriever import index, retriever
        stats = index.describe_index_stats()
        count = stats.get("total_vector_count", 0)
        report(OK if count else FAIL, f"index {INDEX_NAME!r}",
               f"{count} vectors" if count else "empty; run build_index.py --rebuild")

        if count:
            docs = retriever.invoke("What is reinforcement learning?")
            report(OK if docs else FAIL, "retrieval", f"{len(docs)} documents")
            if docs:
                with_src = [d for d in docs if d.metadata.get("source")]
                report(OK if with_src else FAIL, "citation metadata",
                       f"{len(with_src)}/{len(docs)} carry `source`"
                       + ("" if with_src else "; rebuild to enable citations"))
    except Exception as exc:
        report(FAIL, "Pinecone", str(exc)[:120])

    print()
    if failures:
        print(f"{failures} check(s) failed.\n")
        return 1
    print("All checks passed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
