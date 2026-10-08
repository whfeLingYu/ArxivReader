"""智能关联检索 Pipeline：读 Related Work / References -> 提取种子 -> 标题检索 ->
逐篇快速总结 -> 强/弱相关判定 -> 候选列表（用户选择后加入详细阅读）。

五步（与前端「智能关联检索」页一一对应）：
  collect    从论文正文取 Related Work 与 References 章节（不下载新内容）
  extract    一次 LLM 调用：从两个章节里恢复「关键论文标题」作为检索种子
  search     对每个种子做 arXiv 标题检索（ti: 多词 AND + 本地相似度重排），
             与已判定过的候选去重后，按匹配度取前 N 篇落库
  summarize  对每篇候选跑 `quick_summary`（只喂标题+摘要，成本可控）
  judge      用发起方论文的元信息 + 候选总结做 LLM 判定：强相关保留 / 弱相关丢弃

可中断 / 可续跑：进度与日志写 `related_runs`；每阶段/每篇之前检查 cancel_requested；
已判定过的候选（同一源论文同一 arxiv_id）下次运行直接跳过，不重复花 token。

对外：
  STAGES / STAGE_TITLES                            五步清单（前端展示）
  iter_related_events(...)                         流式产出事件（CLI 与后台线程共用）
  run_related_pipeline(...)                        汇总式执行（后台线程用）
  start_related_background(...)                    启动后台检索
  promote_related_candidate(...)                   候选 -> 正式库：下载全文 + 预写 summary 产物
                                                   + 跑知识网络 pipeline（跳过 summary）
  start_related_promote_background(...)            启动后台「加入详细阅读」
"""

import datetime
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from sqlalchemy.orm import Session

from app import arxiv_service, config_loader, database, pdf_service, settings_store
from app.models import Paper, RelatedCandidate, RelatedRun
from harness import config as harness_config, runner
from harness.paper import PaperInput

from . import arxiv_search, llm_steps, store

logger = logging.getLogger(__name__)

STAGES: Tuple[str, ...] = ("collect", "extract", "search", "summarize", "judge")
STAGE_TITLES = {
    "collect": "读取 Related Work 与 References",
    "extract": "提取检索种子",
    "search": "ArXiv 标题检索",
    "summarize": "候选快速总结",
    "judge": "相关性判定",
}

# 送进各 LLM 调用的材料上限（token / 字符），防止 prompt 失控
_SEED_TOKENS = 1600          # Related Work 摘录的 token 上限
_REFS_TOKENS = 2400          # References 摘录的 token 上限
_SUMMARY_CHARS = 3000        # 候选总结送入判定 prompt 的字符上限

# 跳过 summary 进入知识网络的标准流程：tree -> deep -> graph -> synthesis。
# deep 不能省：graph 的关系抽取以深读笔记为输入（pipeline 的数据契约）；
# summary 因为在关联检索阶段已生成，promote 时预写为产物后自然被跳过。
RELATED_PIPELINE_STEPS: Tuple[str, ...] = ("tree", "deep", "graph", "synthesis")


def _utcnow() -> datetime.datetime:
    return datetime.datetime.utcnow()


# ---------------------------------------------------------------- 输入组装
def _extract_sections(paper: Paper) -> Tuple[str, str]:
    """取 Related Work 与 References 正文（章节识别失败返回空串）。"""
    doc = PaperInput.from_any(paper).document()
    related_text = ""
    refs_text = ""
    if doc.match_section("related_work") is not None:
        related_text = doc.section_text("related_work", max_tokens=_SEED_TOKENS)
    if doc.match_section("references") is not None:
        refs_text = doc.section_text("references", max_tokens=_REFS_TOKENS)
    return related_text, refs_text


def _seed_input(paper: Paper, related_text: str, refs_text: str, max_seeds: int) -> str:
    blocks = [
        "## 论文",
        "- 标题：{}".format(paper.title or paper.arxiv_id),
        "- arXiv：{}".format(paper.arxiv_id),
        "",
        "## Related Work 摘录",
        related_text or "（未识别到该章节）",
        "",
        "## References 摘录",
        refs_text or "（未识别到该章节）",
        "",
        "## 提取要求",
        "- 最多给出 {} 个种子。".format(max_seeds),
    ]
    return "\n".join(blocks)


def _judge_input(paper: Paper, row: RelatedCandidate) -> str:
    summary = (row.summary_text or "").strip()
    if len(summary) > _SUMMARY_CHARS:
        summary = summary[:_SUMMARY_CHARS] + "…（已截断）"
    return "\n".join([
        "## 发起方论文",
        "- 标题：{}".format(paper.title or paper.arxiv_id),
        "- arXiv：{}".format(paper.arxiv_id),
        "- 分类：{}".format(", ".join(paper.categories)),
        "- 摘要：{}".format((paper.abstract or "（未提供）").replace("\n", " ")[:600]),
        "",
        "## 候选论文",
        "- 标题：{}".format(row.title),
        "- arXiv：{}".format(row.arxiv_id),
        "- 出处：{}".format((row.published or "")[:10] or "未知"),
        "- 快速总结：",
        summary or "（未生成）",
    ])


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _clean_seeds(payload: Any, max_seeds: int) -> List[Dict[str, str]]:
    """清洗模型输出的种子：去空 / 去重 / 限长 / 限量。"""
    raw = payload.get("seeds") if isinstance(payload, dict) else None
    seeds: List[Dict[str, str]] = []
    seen: set = set()
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        key = title.lower()
        if len(title) < 8 or key in seen:      # 过短标题检索无意义（且多为噪声）
            continue
        seen.add(key)
        seeds.append({"title": title[:300],
                      "authors": str(item.get("authors") or "")[:120],
                      "reason": str(item.get("reason") or "")[:200]})
        if len(seeds) >= max_seeds:
            break
    return seeds


# ---------------------------------------------------------------- 编排
@dataclass
class RelatedOutcome:
    run_id: Optional[int] = None
    status: str = "done"
    counters: Dict[str, int] = field(default_factory=dict)
    error: str = ""


def _cancelled(db: Session, run: RelatedRun) -> bool:
    db.refresh(run)
    return bool(run.cancel_requested)


def _finish(db: Session, run: RelatedRun, status: str, counters: Dict[str, int],
            *, error: str = "") -> Dict[str, Any]:
    """终结一轮关联检索：必须同时写 status + stage + finished_at。

    （沿用检索 pipeline 的教训：只写 stage 会让前端轮询永不停止。）
    """
    store.update_related_run(db, run, status=status, stage="done", error=error,
                             counters=counters, finished=True)
    return {"type": "final", "run_id": run.id, "status": status,
            "counters": counters, "error": error}


def iter_related_events(db: Session, run: RelatedRun, *,
                        hcfg: Optional[Dict[str, Any]] = None,
                        kcfg: Optional[Dict[str, Any]] = None,
                        settings: Any = None) -> Iterator[Dict[str, Any]]:
    """流式执行关联检索 pipeline：每步产出事件（STAGE_TITLES），最后给 final。

    事件：{"type":"stage","stage":..,"title":..,"status":"running|done|failed","detail":..}
          {"type":"progress","stage":"summarize|judge","done":..,"total":..}
          {"type":"final","run_id":..,"status":..,"counters":{...}}
    """
    hcfg = hcfg or harness_config.get_harness_config()
    kcfg = kcfg or harness_config.get_knowledge_config()
    settings = settings or settings_store.get_llm_settings(db)
    paper = db.get(Paper, run.paper_id)
    if paper is None:
        yield _finish(db, run, "failed", {}, error="论文不存在（可能已被删除）")
        return

    max_seeds = max(1, int(kcfg.get("related_max_seeds") or 6))
    per_seed = max(1, int(kcfg.get("related_title_matches") or 2))
    max_candidates = max(1, int(kcfg.get("related_max_candidates") or 10))
    counters: Dict[str, int] = dict(run.counters)

    def _emit_stage(stage: str, status: str, detail: str = "", **extra: Any) -> Dict[str, Any]:
        record = {"type": "stage", "stage": stage, "title": STAGE_TITLES[stage],
                  "status": status, "detail": detail, **extra}
        store.update_related_run(db, run,
                                 stage=stage if status == "running" else run.stage,
                                 counters=counters)
        store.log_related(db, run, "[{}] {} {}".format(STAGE_TITLES[stage], status, detail).strip())
        return record

    try:
        # Step 1: 读取 Related Work / References
        yield _emit_stage("collect", "running")
        if not (paper.full_text or "").strip():
            raise RuntimeError("论文没有可用的正文：Related Work 与 References 需要从全文提取"
                               "（可先点「↻ 重抽全文」）")
        related_text, refs_text = _extract_sections(paper)
        counters["related_work_chars"] = len(related_text)
        counters["references_chars"] = len(refs_text)
        if not related_text and not refs_text:
            raise RuntimeError("未在正文中识别到 Related Work / References 章节，无法提取检索种子")
        if not related_text:
            store.log_related(db, run, "未识别到 Related Work 章节：仅用 References 提取种子",
                              level="warn")
        if not refs_text:
            store.log_related(db, run, "未识别到 References 章节：仅用 Related Work 提取种子",
                              level="warn")
        yield _emit_stage("collect", "done",
                          "Related Work {} 字 · References {} 字".format(len(related_text),
                                                                        len(refs_text)),
                          related_work_chars=len(related_text),
                          references_chars=len(refs_text))
        if _cancelled(db, run):
            yield _finish(db, run, "cancelled", counters)
            return

        # Step 2: 提取检索种子
        yield _emit_stage("extract", "running")
        out = llm_steps.call_structured("related_seed_extract.md",
                                        _seed_input(paper, related_text, refs_text, max_seeds),
                                        settings, cfg=hcfg)
        if not out.ok:
            raise RuntimeError(out.error or "检索种子提取失败")
        seeds = _clean_seeds(out.data, max_seeds)
        counters["seeds"] = len(seeds)
        for seed in seeds:
            store.log_related(db, run, "种子：{}".format(seed["title"]))
        if not seeds:
            store.log_related(db, run, "模型未提取出可用种子，本轮结束", level="warn")
            yield _emit_stage("extract", "done", "未提取到可用种子", seeds=0,
                              tokens=(out.usage or {}).get("total_tokens"))
            yield _finish(db, run, "done", counters)
            return
        yield _emit_stage("extract", "done", "提取 {} 个检索种子".format(len(seeds)),
                          seeds=len(seeds), tokens=(out.usage or {}).get("total_tokens"))

        # Step 3: 逐个种子做 arXiv 标题检索
        yield _emit_stage("search", "running")
        known_ids = {value[0] for value in
                     db.query(RelatedCandidate.arxiv_id)
                     .filter(RelatedCandidate.source_paper_id == paper.id).all()}
        collected: Dict[str, Tuple[Any, str]] = {}      # arxiv_id -> (hit, seed_title)
        for seed in seeds:
            if _cancelled(db, run):
                yield _finish(db, run, "cancelled", counters)
                return
            try:
                hits = arxiv_search.search_by_title(seed["title"], max_results=per_seed)
            except Exception as exc:  # noqa: BLE001  单种子失败不影响其他
                store.log_related(db, run, "种子检索失败：{}（{}）".format(
                    seed["title"][:60], str(exc)[:120]), level="warn")
                continue
            added = 0
            for hit in hits:
                if hit.arxiv_id == paper.arxiv_id:      # 排除发起方论文自身
                    continue
                old = collected.get(hit.arxiv_id)
                if old is not None:
                    if hit.match_score > old[0].match_score:
                        collected[hit.arxiv_id] = (hit, seed["title"])
                    continue
                collected[hit.arxiv_id] = (hit, seed["title"])
                added += 1
            store.log_related(db, run, "种子「{}」命中 {} 篇（新增 {}）".format(
                seed["title"][:60], len(hits), added))
        counters["found"] = len(collected)
        fresh = [(hit, seed_title) for arxiv_id, (hit, seed_title) in collected.items()
                 if arxiv_id not in known_ids]
        counters["duplicated"] = len(collected) - len(fresh)
        fresh.sort(key=lambda item: -item[0].match_score)
        selected = fresh[:max_candidates]
        rows: List[RelatedCandidate] = []
        for hit, seed_title in selected:
            row, _created = store.upsert_related_candidate(db, source_paper_id=paper.id,
                                                           hit=hit, seed_title=seed_title,
                                                           run_id=run.id)
            store.log_related(db, run, "候选：{} → {:.2f} {}".format(
                hit.arxiv_id, hit.match_score, hit.title[:80]), arxiv_id=hit.arxiv_id)
            rows.append(row)
        counters["selected"] = len(rows)
        yield _emit_stage("search", "done",
                          "命中 {} 篇（去重跳过 {} 篇），取前 {} 篇进入总结".format(
                              len(collected), counters["duplicated"], len(rows)),
                          found=len(collected), selected=len(rows))
        if not rows:
            yield _finish(db, run, "done", counters)
            return
        if _cancelled(db, run):
            yield _finish(db, run, "cancelled", counters)
            return

        # Step 4: 对每篇候选跑快速总结
        yield _emit_stage("summarize", "running",
                          "对 {} 篇候选运行快速总结".format(len(rows)))
        summarized: List[RelatedCandidate] = []
        for index, row in enumerate(rows, start=1):
            if _cancelled(db, run):
                yield _finish(db, run, "cancelled", counters)
                return
            paper_input = PaperInput(arxiv_id=row.arxiv_id, title=row.title,
                                     abstract=row.abstract, authors=row.authors,
                                     categories=row.categories, published=row.published,
                                     full_text="")
            summary = runner.run_skill("quick_summary", paper_input, settings, cfg=hcfg)
            if summary.error or not summary.output.strip():
                store.set_related_verdict(
                    db, row, relevance="weak", score=0.0,
                    reason="快速总结失败（按弱相关丢弃）：{}".format(
                        (summary.error or "输出为空")[:120]))
                counters["failed"] = counters.get("failed", 0) + 1
                store.log_related(db, run, "总结失败（按弱相关丢弃）：{}".format(row.arxiv_id),
                                  level="error", arxiv_id=row.arxiv_id)
            else:
                store.set_related_summary(db, row, summary_text=summary.output,
                                          stats={**(summary.stats or {}), "seed": row.seed_title})
                counters["summarized"] = counters.get("summarized", 0) + 1
                summarized.append(row)
                store.log_related(db, run, "已总结：{}".format(row.arxiv_id),
                                  arxiv_id=row.arxiv_id)
            yield {"type": "progress", "stage": "summarize", "done": index,
                   "total": len(rows), "arxiv_id": row.arxiv_id,
                   "counters": dict(counters)}
        yield _emit_stage("summarize", "done",
                          "完成总结 {} 篇（失败 {} 篇）".format(len(summarized),
                                                             counters.get("failed", 0)),
                          summarized=len(summarized))

        # Step 5: 强/弱相关判定
        yield _emit_stage("judge", "running",
                          "对 {} 篇候选做相关性判定".format(len(summarized)))
        for index, row in enumerate(summarized, start=1):
            if _cancelled(db, run):
                yield _finish(db, run, "cancelled", counters)
                return
            try:
                verdict = llm_steps.call_structured("related_judge.md",
                                                    _judge_input(paper, row),
                                                    settings, cfg=hcfg)
            except Exception as exc:  # noqa: BLE001  单篇失败不影响整轮
                verdict = llm_steps.StepOutput(ok=False, error=str(exc)[:200])
            if not verdict.ok:
                store.set_related_verdict(db, row, relevance="weak", score=0.0,
                                          reason="判定失败（按弱相关丢弃）：{}".format(
                                              (verdict.error or "")[:120]))
                counters["weak"] = counters.get("weak", 0) + 1
                store.log_related(db, run, "判定失败（按弱相关丢弃）：{}".format(row.arxiv_id),
                                  level="error", arxiv_id=row.arxiv_id)
            else:
                payload = verdict.data if isinstance(verdict.data, dict) else {}
                related = bool(payload.get("related"))
                score = _clamp_score(payload.get("score"))
                reason = str(payload.get("reason") or "").strip()
                store.set_related_verdict(db, row,
                                          relevance="strong" if related else "weak",
                                          score=score, reason=reason)
                if related:
                    counters["strong"] = counters.get("strong", 0) + 1
                    store.log_related(db, run, "强相关：{}（{:.2f}）{}".format(
                        row.arxiv_id, score, reason[:80]), arxiv_id=row.arxiv_id)
                else:
                    counters["weak"] = counters.get("weak", 0) + 1
                    store.log_related(db, run, "弱相关丢弃：{}（{}）".format(
                        row.arxiv_id, reason[:80]), level="warn", arxiv_id=row.arxiv_id)
            yield {"type": "progress", "stage": "judge", "done": index,
                   "total": len(summarized), "arxiv_id": row.arxiv_id,
                   "counters": dict(counters)}

        yield _emit_stage("judge", "done",
                          "判定完成：强相关 {} 篇 / 弱相关丢弃 {} 篇".format(
                              counters.get("strong", 0), counters.get("weak", 0)),
                          strong=counters.get("strong", 0), weak=counters.get("weak", 0))
        yield _finish(db, run, "done", counters)
        return
    except Exception as exc:  # noqa: BLE001  整轮失败也要留下账本
        logger.warning("关联检索 pipeline 失败（run=%s）：%s", run.id, str(exc)[:300])
        store.log_related(db, run, "失败：{}".format(str(exc)[:200]), level="error")
        yield _finish(db, run, "failed", counters, error=str(exc)[:500])


def run_related_pipeline(db: Session, run: RelatedRun, *,
                         hcfg: Optional[Dict[str, Any]] = None,
                         kcfg: Optional[Dict[str, Any]] = None,
                         settings: Any = None) -> RelatedOutcome:
    """汇总式执行（后台线程 / 测试）：消费事件流并返回结果对象。

    三重兜底：事件流没有 final 帧时置 failed，绝不让 run 永久停在 running。
    """
    outcome = RelatedOutcome(run_id=run.id)
    got_final = False
    for event in iter_related_events(db, run, hcfg=hcfg, kcfg=kcfg, settings=settings):
        if event.get("type") == "final":
            got_final = True
            outcome.status = event.get("status", "done")
            outcome.counters = event.get("counters") or {}
            outcome.error = event.get("error") or ""
    if not got_final:
        outcome.status = "failed"
        outcome.error = "关联检索流程异常结束（未收到最终事件）"
    # error 传 None = 不覆写：失败原因已在 _finish 写过，不能被空字符串抹掉
    store.update_related_run(db, run, status=outcome.status, error=outcome.error or None,
                             counters=outcome.counters, finished=True)
    return outcome


# ---------------------------------------------------------------- 后台线程
def start_related_background(paper_id: int, *, run_id: Optional[int] = None) -> int:
    """在后台线程里跑关联检索（不阻塞前端）；返回 run_id。

    线程内使用独立 DB 会话；异常统一落到 related_runs.error，前端轮询即可看到。
    """
    db = database.SessionLocal()
    try:
        run = store.get_related_run(db, run_id) if run_id else None
        if run is None:
            run = store.create_related_run(db, paper_id)
        run_id = run.id
    finally:
        db.close()

    def _worker() -> None:
        worker_db = database.SessionLocal()
        try:
            run = store.get_related_run(worker_db, run_id)
            if run is None:
                return
            run_related_pipeline(worker_db, run)
        except Exception as exc:  # noqa: BLE001
            logger.warning("关联检索后台任务异常（run=%s）：%s", run_id, str(exc)[:300])
            try:
                run = store.get_related_run(worker_db, run_id)
                if run is not None:
                    store.update_related_run(worker_db, run, status="failed",
                                             error=str(exc)[:500], finished=True)
            except Exception:  # noqa: BLE001
                pass
        finally:
            worker_db.close()

    threading.Thread(target=_worker, name="related-run-{}".format(run_id), daemon=True).start()
    return run_id


# ---------------------------------------------------------------- 候选 -> 正式库（加入详细阅读）
def promote_related_candidate(db: Session, row: RelatedCandidate, *,
                              settings: Any = None,
                              trigger_pipeline: bool = True) -> Dict[str, Any]:
    """把强相关候选迁入正式库：下载全文 -> 建 Paper -> 预写 summary 产物 -> 触发知识网络。

    流程衔接的关键：候选在检索阶段已有 quick_summary，这里把它预写为 paper_artifacts
    的 summary 产物，再按 `tree -> deep -> graph -> synthesis` 运行 pipeline，
    因此「跳过 summary 步骤」与现有存储/图谱能力完全复用（幂等、可断点续跑）。
    """
    settings = settings or settings_store.get_llm_settings(db)
    existing = db.query(Paper).filter(Paper.arxiv_id == row.arxiv_id).first()
    if existing is None:
        meta = arxiv_service.fetch_metadata(row.arxiv_id)
        folder = database.ensure_inbox(db)      # 先落 Inbox；tree 步骤会把它移入方向文件夹
        dest_dir = os.path.join(config_loader.get_pdf_dir(),
                                arxiv_service.safe_dirname(folder.name))
        pdf_path = arxiv_service.download_pdf(meta, dest_dir)
        text, truncated = pdf_service.extract_text_for_context(str(pdf_path))
        existing = Paper(arxiv_id=meta.arxiv_id, title=meta.title or row.title,
                         authors_json=json.dumps(meta.authors or row.authors, ensure_ascii=False),
                         abstract=meta.abstract or row.abstract,
                         categories_json=json.dumps(meta.categories or row.categories,
                                                    ensure_ascii=False),
                         published=meta.published or row.published,
                         pdf_url=meta.pdf_url, pdf_path=str(pdf_path), folder_id=folder.id,
                         full_text=text, text_truncated=truncated)
        db.add(existing)
        db.commit()
        db.refresh(existing)
    if (row.summary_text or "").strip():
        stats = row.stats or {}
        store.upsert_artifact(db, existing.id, store.ARTIFACT_SUMMARY, row.summary_text,
                              stats=stats, model=str(stats.get("model") or ""),
                              reasoning_effort=stats.get("reasoning_effort"))
    row.status = store.RELATED_PROMOTED
    row.promote_state = "done"
    row.promote_error = ""
    row.promoted_paper_id = existing.id
    row.updated_at = _utcnow()
    db.commit()

    result: Dict[str, Any] = {"paper_id": existing.id, "arxiv_id": existing.arxiv_id,
                              "title": existing.title, "steps": list(RELATED_PIPELINE_STEPS)}
    if trigger_pipeline:
        from . import pipeline as knowledge_pipeline
        outcome = knowledge_pipeline.run_pipeline(db, existing, steps=RELATED_PIPELINE_STEPS,
                                                  settings=settings)
        result["pipeline"] = {"run_id": outcome.run_id, "status": outcome.status,
                              "steps": outcome.steps}
        if outcome.status != "done":            # 已入库但流程未完成：把原因留给前端提示
            row.promote_error = "已入库，但知识网络流程{}：{}".format(
                "被终止" if outcome.status == "cancelled" else "未完成",
                (outcome.error or "")[:300])
            db.commit()
    return result


def _set_related_promote_state(db: Session, row: RelatedCandidate, state: str,
                               error: str = "") -> None:
    row.promote_state = state
    row.promote_error = (error or "")[:1000]
    row.updated_at = _utcnow()
    db.commit()


def start_related_promote_background(candidate_id: int, *,
                                     trigger_pipeline: bool = True) -> str:
    """后台「加入详细阅读」（下载全文 + 建 Paper + 跑知识网络）；返回初始状态。

    前端通过 `GET /api/related/candidates/{id}` 的 `promote_state` 轮询进度
    （queued → running → done/failed，失败原因看 promote_error）。
    """
    db = database.SessionLocal()
    try:
        row = store.get_related_candidate(db, candidate_id)
        if row is None:
            raise ValueError("候选记录不存在：id={}".format(candidate_id))
        if row.status == store.RELATED_PROMOTED and row.promoted_paper_id:
            return row.promote_state or "done"
        if row.promote_state in ("queued", "running"):
            return row.promote_state
        _set_related_promote_state(db, row, "queued")
    finally:
        db.close()

    def _worker() -> None:
        worker_db = database.SessionLocal()
        try:
            row = store.get_related_candidate(worker_db, candidate_id)
            if row is None:
                return
            _set_related_promote_state(worker_db, row, "running")
            promote_related_candidate(worker_db, row, trigger_pipeline=trigger_pipeline)
        except Exception as exc:  # noqa: BLE001  失败写回状态，前端可见
            logger.warning("关联候选提升入库失败（candidate=%s）：%s", candidate_id, str(exc)[:300])
            try:
                row = store.get_related_candidate(worker_db, candidate_id)
                if row is not None:
                    _set_related_promote_state(worker_db, row, "failed", str(exc))
            except Exception:  # noqa: BLE001
                pass
        finally:
            worker_db.close()

    threading.Thread(target=_worker, name="related-promote-{}".format(candidate_id),
                     daemon=True).start()
    return "queued"
