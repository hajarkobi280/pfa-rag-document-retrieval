import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
import app as module
import pdf_index
from local_storage import file_lock

A, B, C = "a" * 32, "b" * 32, "c" * 32


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    history = tmp_path / "history"
    history.mkdir()
    monkeypatch.setattr(module, "HISTORY_DIR", history)
    monkeypatch.setattr(pdf_index, "list_documents", lambda: [])
    monkeypatch.setattr(module, "is_administrative", lambda *a: "administrative")
    monkeypatch.setattr(module, "retrieve_service_public_result", lambda *a: {"reranked": []})
    monkeypatch.setattr(module, "_internet_result", lambda *a: {"status": "answered", "claims": [], "sources": [], "source_used": "internet"})
    module.app.config["TESTING"] = True
    return module.app.test_client()


def save(sid, question="Bonjour"):
    module.save_history(sid, [{"role": "user", "content": question, "timestamp": "2026-10-25T10:00:00"}])


def test_lazy_creation_and_legacy_list(client):
    sid = client.post("/api/session").json["session_id"]
    assert not module._history_path(sid).exists()
    module._history_path(A).write_text(json.dumps([{"role": "user", "content": "X" * 75, "timestamp": "2026-10-24T10:00:00"}]), encoding="utf-8")
    module._history_path(B).write_text(json.dumps([{"role": "user", "content": "Plus récent", "timestamp": "2026-10-25T10:00:00"}]), encoding="utf-8")
    module._history_path(C).write_text("[]", encoding="utf-8")
    (module.HISTORY_DIR / "not-a-session.json").write_text("{}", encoding="utf-8")
    rows = client.get("/api/sessions").json
    assert [r["session_id"] for r in rows] == [B, A]
    assert rows[1]["title"] == "X" * 50
    assert rows[0]["message_count"] == 1 and rows[0]["doc_count"] == 0
    assert client.get("/api/history/" + A).json["history"][0]["role"] == "user"


def test_rename_and_delete(client):
    save(A)
    response = client.post(f"/api/sessions/{A}/rename", json={"title": "  <b>Un titre</b>\n" + "x" * 100})
    assert response.status_code == 200
    assert response.json["title"].startswith("Un titre ")
    assert len(response.json["title"]) == 80
    assert client.post(f"/api/sessions/{A}/rename", json={"title": "\n  "}).status_code == 400
    assert client.get("/api/sessions").json[0]["title"] == response.json["title"]
    assert client.post(f"/api/sessions/{A}/delete").status_code == 200
    assert not module._history_path(A).exists()


def test_upload_shared_pdf_and_reference_counting(client, monkeypatch):
    docs = []
    def ingest(path, title):
        if docs: return "PDF_shared", 12, "déjà indexé"
        docs.append({"doc_id": "PDF_shared", "filename": title, "chunks": 12})
        return "PDF_shared", 12, "indexé"
    monkeypatch.setattr(pdf_index, "ingest", ingest)
    monkeypatch.setattr(pdf_index, "list_documents", lambda: docs)
    removed = []
    monkeypatch.setattr(pdf_index, "delete_document", lambda doc: removed.append(doc))
    for sid in (A, B):
        response = client.post("/api/upload", data={"session_id": sid, "file": (io.BytesIO(b"%PDF-fake"), "partage.pdf")})
        assert response.status_code == 200
        assert response.json["event"]["role"] == "event"
        assert module.load_history(sid)[0]["role"] == "event"
    assert len(list((module.ROOT / "data/uploads").glob("*.pdf"))) == 1
    assert client.get("/api/sessions").json[0]["doc_count"] == 1
    assert client.post(f"/api/sessions/{A}/delete").status_code == 200
    assert removed == []
    assert client.post(f"/api/sessions/{B}/delete").status_code == 200
    assert removed == ["PDF_shared"]


def test_search_only_conversation_docs_and_default_auto(client, monkeypatch):
    docs = [{"doc_id": "PDF_a", "filename": "a.pdf"}, {"doc_id": "PDF_b", "filename": "b.pdf"}]
    monkeypatch.setattr(pdf_index, "list_documents", lambda: docs)
    module._save_meta({A: {"doc_ids": ["PDF_a"]}, B: {"doc_ids": ["PDF_b"]}})
    calls = []
    monkeypatch.setattr(pdf_index, "search", lambda *a, **kw: calls.append(kw) or [])
    response = client.post("/api/chat", json={"session_id": A, "question": "Comment demander les APL ?"})
    assert response.status_code == 200
    assert calls[0]["doc_ids"] == ["PDF_a"]
    calls.clear()
    assert client.post("/api/chat", json={"session_id": C, "question": "Comment demander les APL ?"}).status_code == 200
    assert not calls
    assert client.post("/api/chat", json={"session_id": A, "doc_id": "PDF_b", "question": "Quel montant ?"}).status_code == 400


def test_multiple_docs_search_all_explain_latest_or_named(client, monkeypatch):
    docs = [{"doc_id": "PDF_a", "filename": "premier.pdf"}, {"doc_id": "PDF_b", "filename": "dernier.pdf"}]
    monkeypatch.setattr(pdf_index, "list_documents", lambda: docs)
    module._save_meta({A: {"doc_ids": ["PDF_a", "PDF_b"]}})
    searches, overviews = [], []
    monkeypatch.setattr(pdf_index, "search", lambda *a, **kw: searches.append(kw) or [])
    monkeypatch.setattr(pdf_index, "get_document_overview", lambda doc_id, **kw: overviews.append(doc_id) or [])
    assert client.post("/api/chat", json={"session_id": A, "question": "Quels montants sont indiqués ?"}).status_code == 200
    assert searches[0]["doc_ids"] == ["PDF_a", "PDF_b"] and searches[0]["doc_id"] is None
    for question in ("explique le document", "explique le document premier.pdf"):
        assert client.post("/api/chat", json={"session_id": A, "question": question}).status_code == 200
    assert overviews == ["PDF_b", "PDF_a"]


@pytest.mark.parametrize("sid", ["invalid", "..%2Foutside", "A" * 32])
def test_invalid_session_paths(client, sid):
    assert client.get("/api/history/" + sid).status_code == 400
    assert client.post(f"/api/sessions/{sid}/rename", json={"title": "test"}).status_code == 400
    assert client.post(f"/api/sessions/{sid}/delete").status_code == 400
    assert client.post("/api/chat", json={"session_id": sid, "question": "Bonjour"}).status_code == 400
    assert client.post("/api/upload", data={"session_id": sid, "file": (io.BytesIO(b"%PDF-fake"), "a.pdf")}).status_code == 400


def test_sessions_read_does_not_take_generation_lock(client):
    save(A)
    with file_lock(module.ROOT / "data/operation.lock"):
        assert client.get("/api/sessions").status_code == 200
        assert client.post("/api/chat", json={"question": "Bonjour"}).status_code == 409


def test_pdf_search_doc_ids_is_backward_compatible(monkeypatch):
    rows = [{"chunk_id": "PDF_a_0", "doc_id": "PDF_a", "chunk_text": "A"}, {"chunk_id": "PDF_b_0", "doc_id": "PDF_b", "chunk_text": "B"}]
    monkeypatch.setattr(pdf_index, "_read_state", lambda: ({}, rows, np.array([[0.8], [0.9]])))
    monkeypatch.setattr(pdf_index, "_load_model", lambda: SimpleNamespace(encode=lambda *a, **k: np.array([[1.0]])))
    assert [r["doc_id"] for r in pdf_index.search("question", doc_ids=["PDF_a"])] == ["PDF_a"]
    assert [r["doc_id"] for r in pdf_index.search("question", doc_id="PDF_b")] == ["PDF_b"]
    assert len(pdf_index.search("question")) == 2
    assert pdf_index.search("question", doc_ids=[]) == []


def test_detach_shared_pdf_and_explain_scope(client, monkeypatch):
    monkeypatch.setattr(pdf_index, "list_documents", lambda: [{"doc_id": "PDF_shared", "filename": "shared.pdf"}])
    module._save_meta({A: {"doc_ids": ["PDF_shared"]}, B: {"doc_ids": ["PDF_shared"]}})
    removed = []
    monkeypatch.setattr(pdf_index, "delete_document", lambda doc: removed.append(doc))
    assert client.post("/api/documents/delete", json={"session_id": A, "doc_id": "PDF_shared"}).status_code == 200
    assert removed == []
    assert client.post("/api/documents/explain", json={"session_id": A, "doc_id": "PDF_shared"}).status_code == 400
    assert client.get("/api/documents?session_id=" + A).json["documents"] == []
