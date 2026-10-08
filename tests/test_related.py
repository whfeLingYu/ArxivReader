"""智能关联检索测试（全部离线：arXiv 与 LLM 一律 monkeypatch）。

覆盖：
  - 标题检索工具（build_title_query / title_similarity / search_by_title 过滤排序）
  - 五步 pipeline（种子提取 -> 标题检索 -> 总结 -> 判定）与强弱分档、去重续跑、取消、终态
  - 候选提升入库：预写 summary 产物 + 跳过 summary 的 pipeline 步骤（tree/deep/graph/synthesis）
  - HTTP 路由：启动 / 视图 / 中断 / 丢弃 / 删除 / promote 后台状态
"""

import json
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import arxiv_service, database
from app.database import Base
from app.main import app
from app.models import Paper
from harness import llm
from harness.runner import SkillResult
from knowledge import arxiv_search, related, store
from knowledge.llm_steps import StepOutput

_PAPER_TEXT = (
    "[Page 1]\n"
    "Abstract\nWe study credit assignment in agentic RL.\n"
    "1 Introduction\nGroup-relative advantage helps.\n"
    "2 Related Work\nGRPO (Shao et al., 2024) reduces variance; "
    "PPO (Schulman et al., 2017) is the classic baseline.\n"
    "3 Method\nOur estimator improves GRPO.\n"
    "4 Experiments\nWorks on GSM8K.\n"
    "References\n[1] Shao et al. Group Relative Policy Optimization. 2024.\n"
    "[2] Schulman et al. Proximal Policy Optimization Algorithms. 2017.\n"
)


def _mk_paper(db, arxiv_id="2401.00001", title="Credit Assignment for Agentic RL",
              text=_PAPER_TEXT):
    paper = Paper(arxiv_id=arxiv_id, title=title,
                  abstract="We propose a group-relative credit assignment estimator.",
                  authors_json=json.dumps(["A. Author"], ensure_ascii=False),
                  categories_json=json.dumps(["cs.LG"], ensure_ascii=False),
                  published="2024-01-01T00:00:00", full_text=text)
    db.add(paper)
    db.commit()
    db.refresh(paper)
    return paper


def _hit(arxiv_id, title, score=0.9):
    hit = arxiv_search.SearchHit(arxiv_id=arxiv_id, title=title,
                                 abstract="A related paper about credit assignment.",
                                 authors=["B. Author"], categories=["cs.LG"],
                                 published="2024-05-01T00:00:00")
    hit.match_score = score
    return hit


@pytest.fixture()
def db(tmp_path, monkeypatch):
    engine = create_engine("sqlite:///" + str(tmp_path / "related.db"),
                           connect_args={"check_same_thread": False}, future=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    session = Session()
    # 图谱 md 落到临时目录，避免污染 data/
    monkeypatch.setattr("app.config_loader.get_base_dir", lambda: str(tmp_path))
    try:
        yield session
    finally:
        session.close()


# ---------------------------------------------------------------- 标题检索工具
def test_build_title_query_cleans_stopwords_and_dedupes():
    query = arxiv_search.build_title_query("Attention Is All You Need")
    assert "ti:attention" in query and "ti:you" in query and "ti:need" in query
    assert "ti:the" not in query and "ti:all" not in query       # 停用词被剔除
    assert arxiv_search.build_title_query("Test Test Paper").count("ti:") == 2   # 去重
    assert arxiv_search.build_title_query("A An The") == ""      # 全停用词 -> 空
    assert arxiv_search.build_title_query("") == ""


def test_title_similarity_orders_matches():
    base = "Group Relative Policy Optimization"
    near = "Group Relative Policy Optimization for LLMs"
    far = "Vision Transformers for Object Detection"
    assert arxiv_search.title_similarity(base, near) > \
        arxiv_search.title_similarity(base, far)
    assert arxiv_search.title_similarity(base, far) < arxiv_search._TITLE_SIMILARITY_FLOOR


class _FakeResult:
    def __init__(self, short_id, title):
        self._id = short_id
        self.title = title
        self.summary = "abstract"
        self.authors = []
        self.categories = []
        self.published = None
        self.updated = None
        self.comment = ""

    def get_short_id(self):
        return self._id


def test_search_by_title_filters_and_sorts(monkeypatch):
    results = [
        _FakeResult("2402.00002", "Vision Transformers for Object Detection"),
        _FakeResult("2402.00001", "Group Relative Policy Optimization"),
    ]

    class _FakeClient:
        def __init__(self, **kwargs):
            pass

        def results(self, search):
            return list(results)

    monkeypatch.setattr(arxiv_search.arxiv, "Client", lambda **kw: _FakeClient())
    hits = arxiv_search.search_by_title("Group Relative Policy Optimization", max_results=2)
    assert [h.arxiv_id for h in hits] == ["2402.00001"]          # 低相似度被过滤
    assert hits[0].match_score >= arxiv_search._TITLE_SIMILARITY_FLOOR


# ---------------------------------------------------------------- pipeline 编排
_SEEDS = {"seeds": [
    {"title": "Group Relative Policy Optimization", "authors": "Shao 等", "reason": "同问题基线"},
    {"title": "Proximal Policy Optimization Algorithms", "authors": "Schulman 等",
     "reason": "经典基线"},
]}


def _patch_related(monkeypatch, *, hits_by_seed=None, judge_related=True, seeds=None):
    calls = {"search": [], "summary": 0, "judge": 0}

    def fake_search_by_title(title, *, max_results=3, client=None):
        calls["search"].append(title)
        return list((hits_by_seed or {}).get(title, []))

    def fake_skill(skill, paper, settings, **kw):
        calls["summary"] += 1
        return SkillResult(skill="quick_summary",
                           output="TL;DR: x\n- 解决了什么: 用 group-relative 估计降方差\n"
                                  "- 未来展望: 扩展到多智能体",
                           reasoning_content="思考",
                           stats={"usage": {"total_tokens": 30}, "model": "m",
                                  "reasoning_effort": "max"})

    def fake_call(prompt_name, user_text, settings, *, as_json=True, cfg=None,
                  json_retries=2, should_stop=None):
        if prompt_name == "related_seed_extract.md":
            return StepOutput(ok=True, data=(seeds if seeds is not None else _SEEDS),
                              usage={"total_tokens": 40})
        calls["judge"] += 1
        return StepOutput(ok=True, data={"related": judge_related, "score": 0.88,
                                         "reason": "同一子问题（credit assignment）"},
                          usage={"total_tokens": 20})

    monkeypatch.setattr(related.arxiv_search, "search_by_title", fake_search_by_title)
    monkeypatch.setattr(related.runner, "run_skill", fake_skill)
    monkeypatch.setattr(related.llm_steps, "call_structured", fake_call)
    return calls


_KCFG = {"related_max_seeds": 6, "related_title_matches": 2, "related_max_candidates": 10}


def test_related_pipeline_marks_strong_and_weak(db, monkeypatch):
    paper = _mk_paper(db)
    hits = {"Group Relative Policy Optimization":
            [_hit("2402.10001", "Group Relative Policy Optimization", 0.95)]}
    calls = _patch_related(monkeypatch, hits_by_seed=hits)

    run = store.create_related_run(db, paper.id)
    outcome = related.run_related_pipeline(db, run, settings=llm.LLMSettings(model="m"),
                                           kcfg=_KCFG, hcfg={})
    assert outcome.status == "done"
    assert outcome.counters["seeds"] == 2
    assert outcome.counters["found"] == 1 and outcome.counters["summarized"] == 1
    assert outcome.counters["strong"] == 1 and outcome.counters.get("weak", 0) == 0
    assert calls["search"] == ["Group Relative Policy Optimization",
                               "Proximal Policy Optimization Algorithms"]
    assert calls["summary"] == 1 and calls["judge"] == 1

    rows = store.list_related_candidates(db, paper.id)
    assert len(rows) == 1
    row = rows[0]
    assert row.arxiv_id == "2402.10001" and row.relevance == "strong"
    assert row.status == store.RELATED_PENDING
    assert row.summary_text.startswith("TL;DR")
    assert row.seed_title == "Group Relative Policy Optimization"
    assert row.relevance_reason.startswith("同一子问题")
    # 实时日志与 run 终态
    messages = " | ".join(log["message"] for log in run.logs)
    assert "强相关：2402.10001" in messages
    assert run.status == "done" and run.finished_at is not None


def test_related_pipeline_drops_weak(db, monkeypatch):
    paper = _mk_paper(db)
    hits = {"Group Relative Policy Optimization":
            [_hit("2402.30001", "Group Relative Policy Optimization", 0.9)]}
    _patch_related(monkeypatch, hits_by_seed=hits, judge_related=False)
    run = store.create_related_run(db, paper.id)
    outcome = related.run_related_pipeline(db, run, settings=llm.LLMSettings(model="m"),
                                           kcfg=_KCFG, hcfg={})
    assert outcome.status == "done" and outcome.counters["weak"] == 1
    assert store.list_related_candidates(db, paper.id) == []       # 弱相关不出现在列表
    assert store.related_counts(db, paper.id)["discarded"] == 1


def test_related_pipeline_dedupes_known_candidates(db, monkeypatch):
    """第二轮：已判定过的候选直接跳过，不重复总结/判定（省 token）。"""
    paper = _mk_paper(db)
    hits = {"Group Relative Policy Optimization":
            [_hit("2402.40001", "Group Relative Policy Optimization", 0.9)]}
    _patch_related(monkeypatch, hits_by_seed=hits)
    first = store.create_related_run(db, paper.id)
    related.run_related_pipeline(db, first, settings=llm.LLMSettings(model="m"),
                                 kcfg=_KCFG, hcfg={})

    calls = _patch_related(monkeypatch, hits_by_seed=hits)
    second = store.create_related_run(db, paper.id)
    outcome = related.run_related_pipeline(db, second, settings=llm.LLMSettings(model="m"),
                                           kcfg=_KCFG, hcfg={})
    assert outcome.counters["found"] == 1 and outcome.counters["duplicated"] == 1
    assert outcome.counters.get("summarized", 0) == 0
    assert calls["summary"] == 0 and calls["judge"] == 0
    assert len(store.list_related_candidates(db, paper.id)) == 1   # 不重复建行


def test_related_pipeline_cancel_converges(db, monkeypatch):
    paper = _mk_paper(db)
    _patch_related(monkeypatch)
    run = store.create_related_run(db, paper.id)
    run.cancel_requested = True
    db.commit()
    outcome = related.run_related_pipeline(db, run, settings=llm.LLMSettings(model="m"),
                                           kcfg=_KCFG, hcfg={})
    assert outcome.status == "cancelled"
    assert run.status == "cancelled" and run.finished_at is not None


def test_related_pipeline_fails_without_body(db, monkeypatch):
    paper = _mk_paper(db, text="")
    _patch_related(monkeypatch)
    run = store.create_related_run(db, paper.id)
    outcome = related.run_related_pipeline(db, run, settings=llm.LLMSettings(model="m"),
                                           kcfg=_KCFG, hcfg={})
    assert outcome.status == "failed" and "正文" in outcome.error
    assert run.status == "failed" and run.finished_at is not None


def test_related_pipeline_no_seeds_still_finishes(db, monkeypatch):
    paper = _mk_paper(db)
    calls = _patch_related(monkeypatch, seeds={"seeds": []})
    run = store.create_related_run(db, paper.id)
    outcome = related.run_related_pipeline(db, run, settings=llm.LLMSettings(model="m"),
                                           kcfg=_KCFG, hcfg={})
    assert outcome.status == "done" and outcome.counters["seeds"] == 0
    assert calls["search"] == [] and calls["summary"] == 0
    assert run.status == "done" and run.finished_at is not None


# ---------------------------------------------------------------- 候选 -> 正式库
def test_promote_related_candidate_skips_summary(db, monkeypatch, tmp_path):
    paper = _mk_paper(db)
    row, created = store.upsert_related_candidate(
        db, source_paper_id=paper.id, hit=_hit("2402.50001", "Related paper", 0.9),
        seed_title="Group Relative Policy Optimization")
    assert created is True
    store.set_related_summary(db, row, summary_text="TL;DR: 候选总结",
                              stats={"model": "m", "usage": {"total_tokens": 10}})
    store.set_related_verdict(db, row, relevance="strong", score=0.9, reason="ok")

    meta = arxiv_service.PaperMeta(arxiv_id="2402.50001", title="Related paper (arxiv)",
                                   authors=["B. Author"], abstract="abs", categories=["cs.LG"],
                                   published="2024-05-01", pdf_url="http://example.com/x.pdf")
    pdf_path = tmp_path / "2402.50001.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(related.arxiv_service, "fetch_metadata", lambda arxiv_id: meta)
    monkeypatch.setattr(related.arxiv_service, "download_pdf", lambda m, dest: pdf_path)
    monkeypatch.setattr(related.pdf_service, "extract_text_for_context",
                        lambda path, max_chars=120000: ("Body text.", False))
    called = {}

    class _FakeOutcome:
        run_id = 7
        status = "done"
        steps = [{"step": "tree", "status": "done"}]

    def fake_run_pipeline(db_, paper_, **kw):
        called["paper_id"] = paper_.id
        called.update(kw)
        return _FakeOutcome()

    monkeypatch.setattr("knowledge.pipeline.run_pipeline", fake_run_pipeline)

    result = related.promote_related_candidate(db, row, trigger_pipeline=True)
    # 跳过 summary（复用候选总结）：summary 不在步骤里，deep 保留（graph 的输入契约）
    assert tuple(called["steps"]) == ("tree", "deep", "graph", "synthesis")
    assert "summary" not in called["steps"]
    assert called["paper_id"] == result["paper_id"]
    assert result["pipeline"]["run_id"] == 7
    # 预写 summary 产物（pipeline 的 summary 步骤会因此被幂等跳过）
    artifact = store.get_artifact(db, result["paper_id"], store.ARTIFACT_SUMMARY)
    assert artifact is not None and artifact.content_md.startswith("TL;DR")
    assert artifact.model == "m"
    # 候选状态落位
    db.refresh(row)
    assert row.status == store.RELATED_PROMOTED and row.promote_state == "done"
    assert row.promoted_paper_id == result["paper_id"]


# ---------------------------------------------------------------- API
@pytest.fixture()
def client(tmp_path, monkeypatch):
    engine = create_engine("sqlite:///" + str(tmp_path / "api.db"),
                           connect_args={"check_same_thread": False}, future=True)
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "SessionLocal", TestingSession)
    monkeypatch.setattr("app.config_loader.get_base_dir", lambda: str(tmp_path))
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[database.get_db] = override_get_db
    db = TestingSession()
    try:
        database.ensure_inbox(db)
    finally:
        db.close()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _add_paper(client):
    db = database.SessionLocal()
    try:
        return _mk_paper(db).id
    finally:
        db.close()


def test_api_related_run_view_and_candidates(client, monkeypatch):
    _patch_related(monkeypatch, hits_by_seed={
        "Group Relative Policy Optimization":
            [_hit("2402.60001", "Group Relative Policy Optimization", 0.95)]})
    pid = _add_paper(client)

    empty = client.get("/api/papers/{}/related".format(pid)).json()
    assert empty["run"] is None and empty["candidates"] == []
    # 步骤清单与后端同源（不在前端写死中文名）
    assert [s["name"] for s in empty["stages"]] == list(related.STAGES)
    assert empty["stages"][0]["title"] == related.STAGE_TITLES["collect"]

    r = client.post("/api/papers/{}/related/run".format(pid))
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "running"

    view = None
    for _ in range(60):                       # 后台线程跑 mock 流程，很快结束
        view = client.get("/api/papers/{}/related".format(pid)).json()
        if view["run"] and view["run"]["status"] != "running":
            break
        time.sleep(0.1)
    assert view["run"]["status"] == "done"
    assert view["counts"]["pending"] == 1 and view["counts"]["strong"] == 1
    candidate = view["candidates"][0]
    assert candidate["arxiv_id"] == "2402.60001"
    assert candidate["relevance"] == "strong"
    assert candidate["seed_title"] == "Group Relative Policy Optimization"
    assert candidate["relevance_reason"].startswith("同一子问题")


def test_api_related_candidate_actions(client, monkeypatch):
    pid = _add_paper(client)
    db = database.SessionLocal()
    try:
        row, _ = store.upsert_related_candidate(
            db, source_paper_id=pid, hit=_hit("2402.70001", "Candidate paper", 0.9),
            seed_title="seed")
        store.set_related_summary(db, row, summary_text="TL;DR: y", stats={})
        store.set_related_verdict(db, row, relevance="strong", score=0.8, reason="ok")
        cid = row.id
    finally:
        db.close()

    assert client.get("/api/related/candidates/{}".format(cid)).json()["status"] == "pending"
    monkeypatch.setattr("app.routers.related.related.start_related_promote_background",
                        lambda candidate_id, trigger_pipeline=True: "queued")
    promoted = client.post("/api/related/candidates/{}/promote".format(cid)).json()
    assert promoted["promote_state"] == "queued"

    assert client.post("/api/related/candidates/{}/discard".format(cid)).json()["status"] \
        == "discarded"
    assert client.get("/api/papers/{}/related".format(pid)).json()["candidates"] == []
    assert client.delete("/api/related/candidates/{}".format(cid)).status_code == 204
    assert client.get("/api/related/candidates/{}".format(cid)).status_code == 404


def test_api_related_cancel_requires_record(client):
    pid = _add_paper(client)
    assert client.post("/api/papers/{}/related/cancel".format(pid)).status_code == 404
