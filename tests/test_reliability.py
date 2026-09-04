"""Regression tests for tool reliability, history trimming, relevance and LaTeX.

Self-contained so it needs no test framework:

    python tests/test_reliability.py

Only the tool-timeout case takes real time; nothing here touches the network.
"""
import os
import re
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


# --------------------------------------------------------------------------
# 1. External search: timeout, error, zero results
# --------------------------------------------------------------------------
def test_tools():
    print("\n[search reliability]")
    import src.agent.tools as T
    from src.config import TOOL_TOTAL_DEADLINE

    class Hang:
        name, description = "hang", "never returns"
        def invoke(self, _): time.sleep(60)

    class Boom:
        name, description = "boom", "always raises"
        def invoke(self, _): raise RuntimeError("upstream 500")

    class Empty:
        name, description = "empty", "returns nothing"
        def invoke(self, _): return []

    class Fine:
        name, description = "fine", "works"
        def invoke(self, _): return "real result"

    t0 = time.time()
    out = T.bounded(Hang()).invoke({"query": "x"})
    elapsed = time.time() - t0
    check("timeout is bounded by the deadline",
          elapsed <= TOOL_TOTAL_DEADLINE + 2, f"{elapsed:.1f}s <= {TOOL_TOTAL_DEADLINE}s")
    check("timeout degrades instead of raising", "unavailable" in out)
    check("timeout is recorded for the user-facing notice", "hang" in T.DEGRADED)

    out = T.bounded(Boom()).invoke({"query": "x"})
    check("error degrades instead of raising", "unavailable" in out)
    check("error text reaches the model", "upstream 500" in out)

    out = T.bounded(Empty()).invoke({"query": "x"})
    check("zero results is distinct from failure",
          "no results" in out and "unavailable" not in out, out[:60])

    out = T.bounded(Fine()).invoke({"query": "x"})
    check("healthy tool passes through", out == "real result")
    check("recovery clears the degraded flag", "fine" not in T.DEGRADED)


# --------------------------------------------------------------------------
# 2. History trimming
# --------------------------------------------------------------------------
def test_trim():
    print("\n[history trimming]")
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from src.agent.graph import _trim_history
    from src.config import MAX_TOOL_RESULT_CHARS

    convo = [
        HumanMessage(content="explain agentdojo and its findings"),
        AIMessage(content="", tool_calls=[{"name": "arxiv", "args": {}, "id": "1"}]),
        ToolMessage(content="Error: HTTPError" * 12, tool_call_id="1"),
        AIMessage(content="", tool_calls=[{"name": "tavily", "args": {}, "id": "2"}]),
        ToolMessage(content="X" * 8800, tool_call_id="2"),
    ]
    kept = _trim_history(convo)
    check("question survives an oversized tool payload",
          any(isinstance(m, HumanMessage) for m in kept))
    check("search results survive rather than being dropped",
          any(isinstance(m, ToolMessage) and len(m.content) > 100 for m in kept))
    check("oversized payload is capped",
          all(len(m.content) <= MAX_TOOL_RESULT_CHARS + 40
              for m in kept if isinstance(m, ToolMessage)))

    check("empty history returns empty", _trim_history([]) == [])

    huge = [HumanMessage(content="q"),
            AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "9"}]),
            ToolMessage(content="Z" * 50000, tool_call_id="9")]
    check("50k payload still keeps the question",
          any(isinstance(m, HumanMessage) for m in _trim_history(huge)))

    orphan = [ToolMessage(content="orphan", tool_call_id="missing"),
              HumanMessage(content="hi")]
    check("orphaned tool result is dropped",
          not any(isinstance(m, ToolMessage) for m in _trim_history(orphan)))

    stub = [HumanMessage(content="hi"),
            AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "unanswered"}])]
    check("unanswered tool call is dropped",
          not any(getattr(m, "tool_calls", None) for m in _trim_history(stub)))


# --------------------------------------------------------------------------
# 3. Relevance gate (no network: retriever is stubbed)
# --------------------------------------------------------------------------
def test_relevance():
    print("\n[relevance gate]")
    from langchain_core.documents import Document
    import src.agent.graph as G
    from src.config import RELEVANCE_MIN_SCORE

    def docs(scores):
        return [Document(page_content="text", metadata={"score": s, "source": "p.txt"})
                for s in scores]

    real = G.retriever
    try:
        # Off-topic: every neighbour is a weak match.
        G.retriever = types.SimpleNamespace(invoke=lambda q: docs([0.53, 0.50, 0.49, 0.49]))
        out = G.summarizer_node({"messages": [], "refined_query": "off topic"})
        check("off-topic query yields no citations", out["citations"] == [])
        check("off-topic query yields no context", not out["context"])

        # All-zero scores must not divide by zero or slip through.
        G.retriever = types.SimpleNamespace(invoke=lambda q: docs([0.0, 0.0, 0.0, 0.0]))
        out = G.summarizer_node({"messages": [], "refined_query": "junk"})
        check("zero scores are gated out", out["citations"] == [])

        # No documents at all.
        G.retriever = types.SimpleNamespace(invoke=lambda q: [])
        out = G.summarizer_node({"messages": [], "refined_query": "nothing"})
        check("empty retrieval is handled", out["citations"] == [] and not out["context"])

        # Retrieval failure must not take the turn down.
        def boom(q): raise RuntimeError("pinecone down")
        G.retriever = types.SimpleNamespace(invoke=boom)
        out = G.summarizer_node({"messages": [], "refined_query": "x"})
        check("retrieval failure degrades gracefully", out["citations"] == [])

        check("threshold is on the mean, documented", 0.0 < RELEVANCE_MIN_SCORE < 1.0)
    finally:
        G.retriever = real


# --------------------------------------------------------------------------
# 3b. Query splitting (no network: the utility model is stubbed)
# --------------------------------------------------------------------------
def test_split():
    print("\n[query splitting]")
    import src.agent.graph as G
    from langchain_core.messages import AIMessage, HumanMessage

    real = G.llm_utility

    def stub(reply):
        return types.SimpleNamespace(invoke=lambda p: AIMessage(content=reply))

    msg = {"messages": [HumanMessage(content="What is RL, and who won the 2024 F1 title?")]}
    try:
        G.llm_utility = stub('{"ml": ["What is RL?"], "general": ["who won the 2024 F1 title?"]}')
        out = G.classify_node(msg)
        check("mixed message splits into both lists",
              out["ml_parts"] == ["What is RL?"] and
              out["general_parts"] == ["who won the 2024 F1 title?"])

        G.llm_utility = stub('```json\n{"ml": ["What is RL?"], "general": []}\n```')
        out = G.classify_node(msg)
        check("fenced json is parsed", out["ml_parts"] == ["What is RL?"])

        G.llm_utility = stub('Sure! {"ml": [], "general": ["hi"]} hope that helps')
        out = G.classify_node(msg)
        check("json embedded in prose is parsed", out["general_parts"] == ["hi"])

        # Unparseable -> falls back to the yes/no classifier, which also stubs to
        # this reply; "no" routes the whole message to the general path.
        G.llm_utility = stub("no")
        out = G.classify_node(msg)
        check("unparseable output falls back to a single label",
              len(out["ml_parts"]) + len(out["general_parts"]) == 1)

        G.llm_utility = stub('{"ml": [], "general": []}')
        out = G.classify_node(msg)
        check("empty split falls back rather than routing nowhere",
              len(out["ml_parts"]) + len(out["general_parts"]) == 1)

        class Boom:
            def invoke(self, p): raise RuntimeError("model down")
        G.llm_utility = Boom()
        out = G.classify_node(msg)
        check("model failure defaults to the retrieval path", out["ml_parts"])

        check("router sends ML parts to retrieval",
              G.route_after_classify({"ml_parts": ["x"], "general_parts": []}) == "QueryFramer")
        check("router sends general-only straight to the chatbot",
              G.route_after_classify({"ml_parts": [], "general_parts": ["x"]}) == "chatbot")
        check("router sends mixed to retrieval first",
              G.route_after_classify({"ml_parts": ["x"], "general_parts": ["y"]}) == "QueryFramer")
    finally:
        G.llm_utility = real


# --------------------------------------------------------------------------
# 3c. Recovery when the model emits an unoffered tool call / is rate limited
# --------------------------------------------------------------------------
def test_tool_free_recovery():
    print("\n[tool-free recovery]")
    import src.agent.graph as G
    from langchain_core.messages import AIMessage, HumanMessage

    ERR = ("Error code: 400 - {'error': {'message': 'Tool choice is none, but model "
           "called a tool', 'code': 'tool_use_failed'}}")

    check("groq 400 is classified as a tool error", G._is_tool_error(Exception(ERR)))
    BLOCKED = ("Error code: 403 - {'error': {'message': 'Access denied. "
               "Please check your network settings.'}}")
    check("403 is classified as provider-unreachable", G._is_provider_unreachable(Exception(BLOCKED)))
    check("401 is classified as provider-unreachable",
          G._is_provider_unreachable(Exception("Error code: 401 - invalid api key")))
    check("429 is not provider-unreachable (it is retryable)",
          not G._is_provider_unreachable(Exception("Error code: 429 rate_limit_exceeded")))
    check("a tool 400 is not provider-unreachable", not G._is_provider_unreachable(Exception(ERR)))
    check("429 is classified as a rate limit",
          G._is_rate_limit(Exception("Error code: 429 - rate_limit_exceeded")))
    check("an unrelated error is neither",
          not G._is_tool_error(Exception("401 bad key"))
          and not G._is_rate_limit(Exception("401 bad key")))

    class Bound:
        def invoke(self, p): raise Exception(ERR)

    class Flaky:
        """Rejects the first n tool-free asks, mirroring real sampling variance."""
        def __init__(self, n): self.n, self.calls = n, 0
        def bind_tools(self, t): return Bound()
        def invoke(self, p):
            self.calls += 1
            if self.calls <= self.n: raise Exception(ERR)
            return AIMessage(content="Recovered answer.")

    real = G.get_llm
    state = {"messages": [HumanMessage(content="q")], "context": "", "tool_rounds": 0}
    try:
        for n, label in [(0, "first"), (1, "second"), (2, "third")]:
            llm = Flaky(n)
            G.get_llm = lambda *a, **k: llm
            out = G.chatbot(state)
            check(f"recovers on the {label} tool-free attempt",
                  out["messages"][0].content == "Recovered answer.")

        llm = Flaky(99)
        G.get_llm = lambda *a, **k: llm
        out = G.chatbot(state)
        body = out["messages"][0].content
        check("gives an honest message when every attempt is rejected",
              "wasn't able" in body and "external lookup for this question" not in body)

        # A blocked provider must stop the turn at classify, not degrade through
        # every step and report work that never happened.
        real_util = G.llm_utility
        class Blocked:
            def invoke(self, p): raise Exception(BLOCKED)
        G.llm_utility = Blocked()
        try:
            G.classify_node({"messages": [HumanMessage(content="q")]})
            check("classify fails fast when the provider is blocked", False, "no exception")
        except Exception as e:
            check("classify fails fast when the provider is blocked",
                  G._is_provider_unreachable(e))
        finally:
            G.llm_utility = real_util

        class Unrelated:
            def bind_tools(self, t): return self
            def invoke(self, p): raise RuntimeError("401 bad key")
        G.get_llm = lambda *a, **k: Unrelated()
        try:
            G.chatbot(state)
            check("unrelated errors still propagate", False, "no exception raised")
        except RuntimeError:
            check("unrelated errors still propagate", True)
    finally:
        G.get_llm = real


# --------------------------------------------------------------------------
# 4. LaTeX rendering, including partial delimiters mid-stream
# --------------------------------------------------------------------------
_DISPLAY = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_INLINE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
_CITE_ARTIFACT = re.compile(r"\u3010[^\u3011]{0,40}\u3011")


def normalize_math(text):
    text = _CITE_ARTIFACT.sub("", text)
    text = _DISPLAY.sub(lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n", text)
    return _INLINE.sub(lambda m: f"${m.group(1).strip()}$", text)


def test_query_hygiene():
    """A typographic dash in a generated query weakens sparse retrieval."""
    print("\n[query hygiene]")
    from src.agent.graph import _fold_punctuation

    check("non-breaking hyphen folds to ascii",
          _fold_punctuation("multi\u2011agent") == "multi-agent")
    check("en/em dashes fold to ascii",
          _fold_punctuation("a\u2013b\u2014c") == "a-b-c")
    check("smart quotes fold to ascii",
          _fold_punctuation("\u201cq\u201d \u2019") == '"q" \'')
    check("non-breaking space folds", _fold_punctuation("a\u00a0b") == "a b")
    check("plain text is unchanged", _fold_punctuation("multi-agent") == "multi-agent")
    check("empty input is safe", _fold_punctuation("") == "")


def test_citation_artifacts():
    """Built-in browsing models emit internal markers that mean nothing to a reader."""
    print("\n[citation artifacts]")
    check("marker is stripped",
          normalize_math("fourth title\u30100\u2020content\u3011.") == "fourth title.")
    check("multiple markers stripped",
          "\u3010" not in normalize_math("a\u30101\u3011b\u30102\u3011c"))
    check("half-arrived marker is left alone",
          normalize_math("title\u30100\u2020con") == "title\u30100\u2020con")
    check("ordinary brackets survive", normalize_math("f(x) [1]") == "f(x) [1]")


def test_latex():
    print("\n[latex]")
    check(r"inline \(...\) becomes $...$",
          normalize_math(r"a policy \(\pi(a|s)\)") == r"a policy $\pi(a|s)$")
    check(r"display \[...\] becomes $$...$$",
          "$$" in normalize_math(r"\[ \mathbb{E}_\pi[\sum \gamma^t r_t] \]"))
    check("MDP tuple renders",
          normalize_math(r"\((S, A, T, R, \gamma)\)") == r"$(S, A, T, R, \gamma)$")
    check("multiline display renders", "$$" in normalize_math("\\[\n x^2 \n\\]"))
    check("existing $$ is left alone", normalize_math("$$x^2$$") == "$$x^2$$")
    check("plain text is untouched", normalize_math("no math") == "no math")

    # A delimiter split across chunks must never produce unbalanced output.
    full = r"The return is \[ \sum_{t} \gamma^t r_t \] end."
    bad = [c for c in range(1, len(full) + 1)
           if normalize_math(full[:c]).count("$$") % 2 != 0]
    check("partial delimiters never corrupt the stream", not bad,
          f"unbalanced at cuts {bad[:5]}" if bad else "all cut points balanced")


if __name__ == "__main__":
    test_tools()
    test_trim()
    test_relevance()
    test_split()
    test_tool_free_recovery()
    test_query_hygiene()
    test_citation_artifacts()
    test_latex()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", ", ".join(FAIL))
    sys.exit(1 if FAIL else 0)
