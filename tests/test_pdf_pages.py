"""PDF 分页渲染接口测试：用真实 PyMuPDF 生成临时 PDF（不联网、不依赖假文件）。

覆盖：
  - GET /api/papers/{id}/pages               页数与页面尺寸清单（前端查看器占位用）
  - GET /api/papers/{id}/page/{n}?scale=..   单页 PNG 渲染（滚轮/捏合缩放的图片来源）
  - 页码越界 / PDF 文件缺失的错误路径
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import database
from app.database import Base
from app.main import app
from app.models import Paper

fitz = pytest.importorskip("fitz")          # PyMuPDF 缺失时跳过（不阻塞其他测试）


@pytest.fixture()
def client(tmp_path, monkeypatch):
    engine = create_engine("sqlite:///" + str(tmp_path / "pdfpages.db"),
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
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _make_pdf(path, pages=2):
    doc = fitz.open()
    for index in range(pages):
        page = doc.new_page(width=400, height=600)
        page.insert_text((72, 72), "Page {} of ArxivReader".format(index + 1))
    doc.save(str(path))
    doc.close()
    return path


def _mk_paper(db, pdf_path=None, arxiv_id="2501.00001"):
    paper = Paper(arxiv_id=arxiv_id, title="PDF pages paper",
                  authors_json="[]", categories_json="[]",
                  pdf_path=str(pdf_path) if pdf_path else None)
    db.add(paper)
    db.commit()
    db.refresh(paper)
    return paper


def test_pdf_pages_geometry_and_png_render(client, tmp_path):
    pdf_path = _make_pdf(tmp_path / "p.pdf", pages=2)
    db = database.SessionLocal()
    try:
        pid = _mk_paper(db, pdf_path).id
    finally:
        db.close()

    pages = client.get("/api/papers/{}/pages".format(pid))
    assert pages.status_code == 200, pages.text
    data = pages.json()
    assert data["count"] == 2
    assert abs(data["pages"][0]["width"] - 400) < 1
    assert abs(data["pages"][0]["height"] - 600) < 1
    assert [item["page"] for item in data["pages"]] == [1, 2]

    img = client.get("/api/papers/{}/page/1".format(pid))
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/png"
    assert img.content[:8] == b"\x89PNG\r\n\x1a\n"          # PNG magic
    assert "max-age" in img.headers.get("cache-control", "")

    # scale 参数影响渲染尺寸（同页不同倍率的图不同）
    small = client.get("/api/papers/{}/page/2?scale=0.5".format(pid)).content
    large = client.get("/api/papers/{}/page/2?scale=2".format(pid)).content
    assert len(large) > len(small)

    # 页码越界 -> 404；scale 超范围 -> 422（由 Query 校验拦下）
    assert client.get("/api/papers/{}/page/0".format(pid)).status_code == 404
    assert client.get("/api/papers/{}/page/3".format(pid)).status_code == 404
    assert client.get("/api/papers/{}/page/1?scale=99".format(pid)).status_code == 422


def test_pdf_pages_missing_file_returns_404(client):
    db = database.SessionLocal()
    try:
        pid = _mk_paper(db, None, arxiv_id="2501.00002").id
    finally:
        db.close()
    assert client.get("/api/papers/{}/pages".format(pid)).status_code == 404
    assert client.get("/api/papers/{}/page/1".format(pid)).status_code == 404
