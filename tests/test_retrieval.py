from rag.retrieval import Retriever, rrf_fuse
from rag.store import Hit


def h(cid, score=0.0):
    return Hit(cid, cid.split("#")[0], "Art. 1", f"text {cid}", "u", score)


def test_rrf_rewards_agreement_between_lists():
    vector = [h("a#1"), h("b#1"), h("c#1")]
    lexical = [h("c#1"), h("d#1"), h("a#1")]
    fused = [x.chunk_id for x in rrf_fuse([vector, lexical])]
    assert fused[:2] == ["a#1", "c#1"]  # in both lists
    assert set(fused) == {"a#1", "b#1", "c#1", "d#1"}


def test_rrf_scores_replace_original_scores():
    fused = rrf_fuse([[h("a#1", 0.9)]], k=60)
    assert fused[0].score == 1 / 61


class FakeStore:
    def vector_search(self, emb, k):
        return [h("v#1"), h("both#1"), h("v#2")][:k]

    def lexical_search(self, q, k):
        return [h("both#1"), h("l#1")][:k]


class FakeEmbedder:
    def embed_one(self, text):
        return [0.0]


class ReverseReranker:
    """Prefers later candidates, to prove the reranker's order wins."""

    def score(self, query, texts):
        return [float(i) for i in range(len(texts))]


def test_hybrid_retriever_fuses_both_lists():
    r = Retriever(FakeStore(), FakeEmbedder(), mode="hybrid", top_k=2)
    assert [x.chunk_id for x in r.retrieve("q")][0] == "both#1"


def test_reranker_reorders_the_pool_and_keeps_top_k():
    r = Retriever(FakeStore(), FakeEmbedder(), mode="vector", top_k=2, reranker=ReverseReranker())
    assert [x.chunk_id for x in r.retrieve("q")] == ["v#2", "both#1"]
