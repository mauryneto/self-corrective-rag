from collections import Counter

from rag.bm25 import BM25Index, parse_tsvector


def test_parse_tsvector_counts_positions():
    assert parse_tsvector("'consult':3,9 'psicólog':4 'd''água':2A") == Counter(
        {"consult": 2, "psicólog": 1, "d'água": 1})


def test_rare_terms_outweigh_common_ones():
    common = Counter({"oper": 1, "plan": 1})
    docs = [common + Counter({"psicólog": 1}), common + Counter({"contábil": 3}), common, common, common]
    idx = BM25Index([f"d{i}" for i in range(5)], docs)
    assert idx.search(["oper", "plan", "psicólog"], 1)[0][0] == "d0"


def test_unknown_terms_return_nothing():
    idx = BM25Index(["d0"], [Counter({"a": 1})])
    assert idx.search(["zzz"], 5) == []
