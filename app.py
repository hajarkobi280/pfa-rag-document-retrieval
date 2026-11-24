"""Application web conversationnelle (Flask) pour l'assistant administratif RAG."""
import sys
from pathlib import Path

# Bloquer pyarrow AVANT tout autre import (arrow.dll crashe sur Python 3.14)
_ROOT_EARLY = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT_EARLY))
import block_pyarrow  # noqa: F401  (doit rester en tête, avant flask/numpy/torch)

import argparse
import json
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from functools import lru_cache
import os
import internet_search

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request, render_template, Response, stream_with_context
from rag_observability import CURRENT, Trace, stage, measure, Cancelled
import queue
from werkzeug.exceptions import BadRequest, HTTPException
from local_storage import atomic_write, file_lock
from functools import wraps
import time

import answer_rag
import pdf_index

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


# --------------------------------------------------------------------------
# Petites conversations, sigles, questions hors périmètre
# --------------------------------------------------------------------------
SMALLTALK = re.compile(
    r"^\s*(bonjour|bonsoir|salut|salam|slt|hello|hi|coucou|merci|ça va|ca va)\b(\s+\w+)?[\s!.?]*$",
    re.I,
)

ACRONYMS = {
    "cnss": "Caisse Nationale de Sécurité Sociale",
    "cnie": "Carte Nationale d'Identité Électronique",
    "amo": "Assurance Maladie Obligatoire",
    "caf": "Caisse d'allocations familiales",
    "cpam": "Caisse primaire d'assurance maladie",
    "cerfa": "formulaire Cerfa",
}

# Démarches marocaines : la base Service-Public ne les couvre pas.
ADMIN_KEYWORDS = tuple(x.strip() for x in os.environ.get(
    "RAG_ADMIN_KEYWORDS", "demarche,dossier,carte d identite,cnie,passeport,permis,visa,titre de sejour,nationalite,impot,taxe,allocation,aide,caf,cnss,amo,retraite,chomage,mairie,prefecture,acte de naissance,mariage,etat civil,cerfa,attestation,certificat,logement,loyer,apl,securite sociale,assurance maladie,declaration,inscription,justificatif,formulaire,anfcc,dgi,sarl,auto entrepreneur,entrepreneur,registre commerce,carte electorale"
).split(",") if x.strip())
OUT_KEYWORDS = tuple(x.strip() for x in os.environ.get(
    "RAG_OUT_KEYWORDS", "recette,cuisine,cuisiner,capitale,meteo,blague,poeme,football,coupe du monde,film,chanson,musique"
).split(",") if x.strip())


def _plain(text):
    import unicodedata
    plain = "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", plain).strip()


@stage('classification')
def is_administrative(question, semantic_top_score=None):
    """Détection rapide puis classification locale ; doute = administratif."""
    text = _plain(question)
    if any(_plain(k) in text for k in ADMIN_KEYWORDS):
        return "administrative"
    if any(_plain(k) in text for k in OUT_KEYWORDS):
        return "autre"
    try:
        schema = {"type":"object", "properties":{"domain":{"type":"string","enum":["administrative","autre"]}}, "required":["domain"]}
        response = answer_rag.request_json('/api/chat', {
            'model': answer_rag.MODEL, 'stream': False, 'think': False, 'format': schema,
            'keep_alive': answer_rag.KEEP_ALIVE,
            'options': {'temperature': 0, 'num_ctx': answer_rag.NUM_CTX, 'num_predict': 10},
            'messages': [{'role':'system','content':'Classe la question. Administrative = organisme ou démarche administrative dans tout pays. Sinon autre. Réponds uniquement au JSON demandé.'}, {'role':'user','content':question}],
        }, timeout=90)
        value = json.loads(answer_rag.response_content(response) or '{}').get('domain')
        return value if value in ('administrative','autre') else 'administrative'
    except Exception:
        return 'administrative'


def _question_terms(question):
    stop = {'comment','demander','faire','obtenir','quel','quelle','quels','quelles','est','pour','dans','avec','mon','ma','mes','une','des','les','du','de','la','le','sur','maroc','france','quoi'}
    codes = {'apl','caf','amo','dgi'}
    return {t for t in re.findall(r'[a-z0-9]+', _plain(question)) if (len(t) >= 4 or t in codes) and t not in stop}


def evidence_matches_question(question, rows):
    terms = _question_terms(question)
    if not terms: return True
    text = _plain(' '.join(str(r.get('title',''))+' '+str(r.get('text',r.get('chunk_text',''))) for r in rows))
    return any(re.search(r'\b'+re.escape(term)+r'(?:s|es)?\b', text) for term in terms)

def _official_sites(question):
    text = _plain(question)
    for key, sites in answer_rag.OFFICIAL_SITES.items():
        if _plain(key) in text:
            return sites
    if 'maroc' in text or any(k in text for k in ('cnss','cnie','amo','dgi','anfcc')):
        return answer_rag.OFFICIAL_SITES['maroc']
    return answer_rag.OFFICIAL_SITES['france']


def expand_acronyms(text):
    """Développe les sigles une seule fois (pas si déjà suivis d'une parenthèse)."""
    for key, value in ACRONYMS.items():
        text = re.sub(rf"\b{key}\b(?!\s*\()", f"{key.upper()} ({value})", text, flags=re.I)
    return text


@app.errorhandler(400)
@app.errorhandler(413)
def invalid_request(error):
    return jsonify({"error": "Fichier trop volumineux (20 Mo maximum)." if error.code == 413 else str(error.description)}), error.code


@app.before_request
def validate_input():
    if request.is_json:
        data = request.get_json()
        if not isinstance(data, dict):
            raise BadRequest("Un objet JSON est requis.")
        for key in ("session_id", "question", "mode", "doc_id", "model"):
            if key in data and data[key] is not None and not isinstance(data[key], str):
                raise BadRequest("Champ de la requête invalide.")
        if data.get("session_id"):
            _history_path(data["session_id"])
        if data.get("question") and len(data["question"]) > 2000:
            raise BadRequest("Question trop longue (2000 caractères maximum).")
        if "mode" in data and data["mode"] not in ("auto", "pdf", "hybrid", "service_public", "internet", "web"):
            raise BadRequest("Paramètre de recherche invalide.")


def exclusive_operation(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            with file_lock(ROOT / "data" / "operation.lock", timeout=0):
                return function(*args, **kwargs)
        except TimeoutError:
            return jsonify({"error": "Une opération est déjà en cours. Réessayez après sa fin."}), 409
        except (ValueError, RuntimeError, OSError) as exc:
            app.logger.exception("Échec de l'opération locale")
            return jsonify({"error": str(exc)}), 500
    return wrapped

HISTORY_DIR = ROOT / "data" / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _history_path(session_id):
    if not isinstance(session_id, str) or not re.fullmatch(r"[a-f0-9]{32}", session_id):
        raise BadRequest("Identifiant de session invalide.")
    return HISTORY_DIR / f"{session_id}.json"


def load_history(session_id):
    path = _history_path(session_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_history(session_id, history):
    atomic_write(_history_path(session_id), json.dumps(history, ensure_ascii=False, indent=2).encode("utf-8"))
    with file_lock(HISTORY_DIR / "_meta.lock"):
        meta = _load_meta()
        meta.setdefault(session_id, {})["updated_at"] = datetime.now().astimezone().isoformat()
        _save_meta(meta)


def _load_meta():
    path = HISTORY_DIR / "_meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_meta(meta):
    atomic_write(HISTORY_DIR / "_meta.json", json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))


def _session_documents(session_id):
    _history_path(session_id)
    ids = _load_meta().get(session_id, {}).get("doc_ids", [])
    indexed = {doc["doc_id"]: doc for doc in pdf_index.list_documents()}
    return [indexed[doc_id] for doc_id in ids if doc_id in indexed]


def _detach_documents(session_id, doc_ids):
    with file_lock(HISTORY_DIR / "_meta.lock"):
        meta = _load_meta()
        entry = meta.setdefault(session_id, {})
        entry["doc_ids"] = [d for d in entry.get("doc_ids", []) if d not in doc_ids]
        referenced = {d for value in meta.values() for d in value.get("doc_ids", [])}
        for doc_id in doc_ids:
            if doc_id not in referenced:
                pdf_index.delete_document(doc_id)
        _save_meta(meta)


def _quick_reply(session_id, history, text, status):
    """Réponse immédiate (sans recherche ni LLM), enregistrée dans l'historique."""
    result = {"status": status, "claims": [], "sources": [], "source_used": "none", "domain": "smalltalk" if status == "smalltalk" else "administrative"}
    history.append({
        "role": "assistant",
        "content": text,
        "result": result,
        "timestamp": datetime.now().isoformat(),
    })
    save_history(session_id, history)
    return jsonify({"session_id": session_id, "answer": text, "result": result})


# --------------------------------------------------------------------------
# Recherche Service-Public : moteur chargé UNE seule fois en mémoire
# --------------------------------------------------------------------------
_searcher = None
_searcher_lock = threading.Lock()


def retrieve_service_public(question):
    """Recherche + reclassement avec E5 et le reranker déjà chargés."""
    global _searcher
    with _searcher_lock:
        if _searcher is None:
            from rerank_search import RerankedSearch
            _searcher = RerankedSearch()
        return _searcher.search(question).get("reranked", [])


def retrieve_service_public_result(question):
    global _searcher
    with _searcher_lock:
        if _searcher is None:
            from rerank_search import RerankedSearch
            _searcher = RerankedSearch()
        return _searcher.search(question)


@lru_cache(maxsize=256)
def _retrieve_details_cached(question_norm):
    return retrieve_service_public_result(question_norm)


def _retrieve_cached(question_norm):
    return tuple(_retrieve_details_cached(question_norm).get("reranked", []))


def retrieve_sp_cached(question):
    """Même question = même résultat, sans recalcul."""
    return list(_retrieve_cached(" ".join(question.split())))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    """Disponibilité HTTP indépendante d'Ollama et des modèles."""
    return jsonify({"ok": True, "application": "pfa-rag", "version": "auto-cascade-2026-10-26"})


@app.route("/api/status", methods=["GET"])
def status():
    """Retourne l'état des services et des index."""
    try:
        model = answer_rag.MODEL
        digest = answer_rag.check_model(model)
        ollama_ok = True
    except Exception:
        ollama_ok = False
        model = answer_rag.MODEL
        digest = None

    docs = pdf_index.list_documents()
    return jsonify({
        "ollama_ready": ollama_ok,
        "embedding_available": (ROOT / "models/multilingual-e5-small/download_complete.txt").exists(),
        "reranker_available": (ROOT / "models/reranker-fr/download_complete.json").exists(),
        "retrieval_index_available": all((ROOT / p).exists() for p in (
            "data/index_lexical/manifest.json", "data/index_service_public/manifest.json")),
        "model": model,
        "model_digest": digest,
        "ollama_version": ollama_version(),
        "documents_count": len(docs),
        "total_pdf_chunks": sum(d.get("chunks", 0) for d in docs),
    })


@app.route("/api/session", methods=["POST"])
def new_session():
    sid = uuid.uuid4().hex
    return jsonify({"session_id": sid})


@app.route("/api/history/<path:session_id>", methods=["GET"])
def get_history(session_id):
    return jsonify({"history": load_history(session_id), "documents": _session_documents(session_id)})


@app.get("/api/sessions")
def sessions():
    meta = _load_meta()
    result = []
    for path in HISTORY_DIR.glob("*.json"):
        if not re.fullmatch(r"[a-f0-9]{32}", path.stem):
            continue
        history = load_history(path.stem)
        if not history: continue
        entry = meta.get(path.stem, {})
        first = next((m.get("content", "") for m in history if m.get("role") == "user"), "Nouvelle conversation")
        updated = entry.get("updated_at") or next((m.get("timestamp") for m in reversed(history) if m.get("timestamp")), datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat())
        result.append({"session_id": path.stem, "title": entry.get("title") or first[:50],
                       "updated_at": updated, "message_count": sum(m.get("role") in ("user", "assistant") for m in history),
                       "doc_count": len(entry.get("doc_ids", []))})
    return jsonify(sorted(result, key=lambda r: datetime.fromisoformat(r["updated_at"]).timestamp(), reverse=True))


@app.post("/api/sessions/<path:session_id>/rename")
def rename_session(session_id):
    path = _history_path(session_id)
    title = (request.get_json(silent=True) or {}).get("title")
    if not isinstance(title, str): raise BadRequest("Titre invalide.")
    title = " ".join(re.sub(r"<[^>]*>|[\x00-\x1f\x7f]", " ", title).split())[:80]
    if not title: raise BadRequest("Le titre ne peut pas être vide.")
    with file_lock(HISTORY_DIR / "_meta.lock"):
        if not path.exists(): return jsonify({"error": "Conversation introuvable."}), 404
        meta = _load_meta()
        meta.setdefault(session_id, {})["title"] = title
        _save_meta(meta)
    return jsonify({"title": title})


@app.post("/api/sessions/<path:session_id>/delete")
@exclusive_operation
def delete_session(session_id):
    path = _history_path(session_id)
    _detach_documents(session_id, _load_meta().get(session_id, {}).get("doc_ids", []))
    with file_lock(HISTORY_DIR / "_meta.lock"):
        meta = _load_meta()
        meta.pop(session_id, None)
        path.unlink(missing_ok=True)
        _save_meta(meta)
    return jsonify({"ok": True})


@app.route("/api/reset", methods=["POST"])
@exclusive_operation
def reset():
    data = request.get_json(silent=True) or {}
    sid = data.get("session_id")
    if sid:
        save_history(sid, [])
    return jsonify({"ok": True})


@app.route("/api/upload", methods=["POST"])
@exclusive_operation
def upload():
    session_id = request.form.get("session_id") or uuid.uuid4().hex
    _history_path(session_id)
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier reçu"}), 400
    f = request.files["file"]
    if not f.filename or not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Seuls les fichiers PDF sont acceptés"}), 400
    uploads = ROOT / "data" / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    filename = f.filename.replace("\\", "/").rsplit("/", 1)[-1]
    dest = uploads / (uuid.uuid4().hex + ".pdf")
    f.save(dest)
    try:
        if not dest.read_bytes().startswith(b"%PDF-"):
            dest.unlink(missing_ok=True)
            return jsonify({"error": "Le contenu reçu n’est pas un PDF valide."}), 400
        doc_id, n, ingest_status = pdf_index.ingest(str(dest), title=filename)
        if ingest_status == "déjà indexé":
            dest.unlink(missing_ok=True)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        return jsonify({"error": f"Erreur lors de l'indexation : {exc}"}), 500
    with file_lock(HISTORY_DIR / "_meta.lock"):
        meta = _load_meta()
        ids = meta.setdefault(session_id, {}).setdefault("doc_ids", [])
        if doc_id not in ids: ids.append(doc_id)
        _save_meta(meta)
    event = {"role": "event", "content": f"PDF ajouté : {filename} ({n} passages)", "timestamp": datetime.now().isoformat()}
    save_history(session_id, load_history(session_id) + [event])
    return jsonify({
        "session_id": session_id, "event": event,
        "doc_id": doc_id,
        "filename": filename,
        "chunks": n,
        "status": ingest_status,
        "documents": _session_documents(session_id),
    })


@app.route("/api/documents", methods=["GET"])
def documents():
    sid = request.args.get("session_id")
    return jsonify({"documents": _session_documents(sid) if sid else pdf_index.list_documents()})


@app.route("/api/documents/delete", methods=["POST"])
@exclusive_operation
def delete_document():
    data = request.get_json(silent=True) or {}
    doc_id = data.get("doc_id")
    if not doc_id:
        return jsonify({"error": "doc_id manquant"}), 400
    sid = data.get("session_id")
    if sid:
        _history_path(sid)
        if doc_id not in [d["doc_id"] for d in _session_documents(sid)]:
            raise BadRequest("Ce PDF n'appartient pas à la conversation.")
        _detach_documents(sid, [doc_id])
        return jsonify({"ok": True, "documents": _session_documents(sid)})
    if any(doc_id in entry.get("doc_ids", []) for entry in _load_meta().values()):
        raise BadRequest("Précisez la conversation pour retirer ce PDF.")
    ok = pdf_index.delete_document(doc_id)
    return jsonify({"ok": ok, "documents": pdf_index.list_documents()})


@app.route("/api/documents/clear", methods=["POST"])
@exclusive_operation
def clear_documents():
    if any(entry.get("doc_ids") for entry in _load_meta().values()):
        raise BadRequest("Retirez les PDF depuis leurs conversations.")
    pdf_index.clear_all()
    return jsonify({"ok": True, "documents": []})


@app.route("/api/documents/explain", methods=["POST"])
@exclusive_operation
def explain_document():
    data = request.get_json(silent=True) or {}
    doc_id = data.get("doc_id")
    session_id = data.get("session_id") or uuid.uuid4().hex
    model = data.get("model") or answer_rag.MODEL

    if not doc_id:
        return jsonify({"error": "Aucun PDF n'est rattaché à cette demande."}), 400
    if doc_id not in [d["doc_id"] for d in _session_documents(session_id)]:
        raise BadRequest("Ce PDF n'appartient pas à la conversation.")
    passages = pdf_index.get_document_overview(doc_id, max_chunks=5)
    if not passages:
        return jsonify({"error": "Aucun passage trouvé pour ce document"}), 404

    question = "Explique le contenu et les éléments clés de ce document administratif."
    history = load_history(session_id)
    history.append({
        "role": "user",
        "content": f"Explique-moi le document : {passages[0].get('title', 'Document PDF')}",
        "timestamp": datetime.now().isoformat(),
    })

    try:
        result = answer_rag.document_overview(passages)
        result.update(question=question, doc_id=doc_id, source_used="pdf",
                      source_notice="Source utilisée : PDF")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    answer_text = answer_rag.render(result)
    history.append({
        "role": "assistant",
        "content": answer_text,
        "result": result,
        "timestamp": datetime.now().isoformat(),
    })
    save_history(session_id, history)
    return jsonify({
        "session_id": session_id,
        "answer": answer_text,
        "result": result,
    })


def _document_request(question):
    return bool(re.search(r"\b(?:pdf|mon document|ce document|ce pdf|mon attestation|cette attestation|fichier importe)\b", _plain(question)))


def is_document_overview_request(question):
    text = _plain(question)
    return bool(re.match(
        r"(?:explique(?: moi)?|resume(?: moi)?|resume du|analyse(?: moi)?|de quoi parle|que contient|c est quoi) "
        r"(?:(?:le|ce|mon|un|du) )?(?:pdf|document)\b", text))


_version_cache = (0.0, None)

def ollama_version():
    global _version_cache
    now = time.monotonic()
    if now - _version_cache[0] > 30:
        try:
            version = answer_rag.request_json('/api/version', timeout=2).get('version')
        except Exception:
            version = None
        _version_cache = (now, version)
    return _version_cache[1]


def _execute_chat(emit=None, cancelled=None):
    trace = Trace(emit, cancelled)
    token = CURRENT.set(trace)
    try:
        try:
            response = app.make_response(_chat_core())
        except Cancelled:
            raise
        except HTTPException as exc:
            response = app.make_response((jsonify({'error': exc.description}), exc.code))
        except Exception as exc:
            app.logger.exception('Échec de la réponse')
            response = app.make_response((jsonify({'error': str(exc)}), 500))
        payload = response.get_json()
        if isinstance(payload, dict):
            result = payload.setdefault('result', {})
            result.setdefault('model', os.environ.get('RAG_MODEL', answer_rag.MODEL))
            result['ollama_version'] = ollama_version()
            result['timings'] = trace.result()
            record = {'timings': result['timings'], 'model': result['model'],
                      'ollama_version': result['ollama_version'],
                      'source_used': result.get('source_used', 'none'),
                      'status': result.get('status', 'error')}
            if os.environ.get('RAG_LOG_QUESTIONS') == '1':
                record['question'] = (request.get_json(silent=True) or {}).get('question')
            path = ROOT / 'data' / 'timings.jsonl'
            with file_lock(path.with_suffix('.lock')):
                with path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            sid = payload.get('session_id')
            if sid:
                history = load_history(sid)
                if history and history[-1].get('role') == 'assistant':
                    history[-1]['result'] = result
                    save_history(sid, history)
            response.set_data(app.json.dumps(payload))
        return response
    finally:
        CURRENT.reset(token)


@app.post('/api/chat/stream')
def chat_stream():
    data = request.get_json(silent=True) or {}
    events = queue.Queue()
    cancelled = threading.Event()
    consumed = threading.Event()
    def work():
        try:
            with app.test_request_context('/api/chat', method='POST', json=data):
                with file_lock(ROOT / 'data' / 'operation.lock', timeout=0):
                    response = _execute_chat(events.put, cancelled)
                    events.put({'type': 'error' if response.status_code >= 400 else 'final', **response.get_json()})
                    # Keep exclusivity until the final frame is consumed or client closes.
                    consumed.wait()
        except Cancelled:
            pass
        except Exception as exc:
            events.put({'type': 'error', 'error': str(exc)})
        finally:
            events.put(None)
    @stream_with_context
    def stream():
        threading.Thread(target=work, daemon=True, name='rag-stream').start()
        try:
            while True:
                try:
                    event = events.get(timeout=1)
                except queue.Empty:
                    yield '\n'  # heartbeat detects disconnects during blocking inference
                    continue
                if event is None:
                    break
                yield json.dumps(event, ensure_ascii=False) + '\n'
                if event['type'] in ('final', 'error'):
                    break
        finally:
            cancelled.set()
            consumed.set()
    return Response(stream(), mimetype='application/x-ndjson',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


def _search_local(question, strategy, docs, doc_id, is_explain, relevance_question=None):
    """Recherche locale selon la stratégie choisie, en conservant le singleton E5."""
    sp_rows, pdf_rows, semantic = [], [], None
    use_sp = strategy in ("auto", "hybrid", "service_public")
    use_pdf = strategy in ("auto", "hybrid", "pdf") and bool(docs)
    if strategy == "pdf" and not docs:
        return [], None, "no_pdf"
    if use_pdf and not doc_id and len(docs) == 1:
        doc_id = docs[0]["doc_id"]
    if use_pdf and is_explain and not doc_id:
        doc_id = docs[-1]["doc_id"]
    if use_pdf and doc_id and not any(d["doc_id"] == doc_id for d in docs):
        raise ValueError("Ce document n’existe plus.")
    pdf_min = None
    if not is_explain:
        key = "RAG_PDF_MIN_PDFONLY" if strategy == "pdf" else "RAG_PDF_MIN_HYBRID"
        pdf_min = float(os.environ[key]) if os.environ.get(key) else None
    @stage('pdf_search')
    def do_pdf():
        if not use_pdf:
            return []
        if is_explain:
            return pdf_index.get_document_overview(doc_id, max_chunks=5 if strategy == "pdf" else 3)
        found = pdf_index.search(question, top_k=5 if strategy == "pdf" else 2,
                                 doc_id=doc_id, min_score=pdf_min, doc_ids=[d["doc_id"] for d in docs])
        if strategy != "pdf" and pdf_min is None:
            # No measured cosine threshold is available yet: require a query term
            # in each hybrid PDF passage instead of trusting every top cosine.
            relevant_text = relevance_question or question
            found = [row for row in found if _question_terms(relevant_text)
                     and evidence_matches_question(relevant_text, [row])]
        return found
    @stage('base_search')
    def do_sp():
        if not use_sp or is_explain:
            return []
        return _retrieve_details_cached(" ".join(question.split()))
    if use_pdf and use_sp:
        try: pdf_rows = do_pdf()
        except Exception: pdf_rows = []
        try:
            sp_result = do_sp()
            sp_rows = sp_result.get("reranked", [])
            semantic = sp_result.get("semantic_top_score")
        except Exception:
            if not pdf_rows: raise
    elif use_pdf:
        pdf_rows = do_pdf()
    elif use_sp:
        sp_result = do_sp()
        sp_rows = sp_result.get("reranked", [])
        semantic = sp_result.get("semantic_top_score")
    combined = []
    seen = set()
    for row in pdf_rows + sp_rows:
        cid = row.get("chunk_id")
        if cid not in seen and len(combined) < 5:
            combined.append(row); seen.add(cid)
    return combined, semantic, None


@stage('reranking')
def _rank_web_passages(pages, question):
    """Découpe et classe les pages avec l'instance E5 déjà chargée par le RAG."""
    global _searcher
    if _searcher is None:
        with _searcher_lock:
            if _searcher is None:
                from rerank_search import RerankedSearch
                _searcher = RerankedSearch()
    model = _searcher.engine.model
    passages = []
    query_terms = _question_terms(question)
    for page_no, page in enumerate(pages, 1):
        body = page.get("text", "")
        candidates = []
        start = 0
        while start < len(body):
            # Réserver un tiers du budget de 3000 caractères à chaque page.
            # Sinon deux passages de 1200 éliminent le meilleur passage de la
            # troisième page et sélectionnent parfois son pied de page court.
            excerpt = body[start:start + 1000].strip()
            if excerpt:
                overlap = sum(1 for term in query_terms if re.search(r"\b" + re.escape(term) + r"(?:s|es)?\b", _plain(excerpt)))
                candidates.append((overlap, start, excerpt))
            if start + 1000 >= len(body): break
            start += 850
        # Cap local E5 work while retaining lexical matches and evenly spaced coverage.
        lexical = sorted(candidates, key=lambda x: (-x[0], x[1]))[:20]
        stride = max(1, len(candidates) // 12)
        sampled = candidates[::stride][:12]
        chosen = {(pos, excerpt) for _, pos, excerpt in lexical + sampled}
        for chunk_no, (pos, excerpt) in enumerate(sorted(chosen, key=lambda x: x[0])):
            passages.append({"chunk_id": f"WEB_{page_no}_{chunk_no}", "title": page.get("title", "Page web"),
                "url": page.get("url", ""), "text": excerpt, "scored_excerpt": excerpt,
                "official": bool(page.get("official")), "context": []})
    if not passages:
        return []
    vectors = model.encode(["passage: " + p["text"] for p in passages], normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True)
    query = model.encode(["query: " + question], normalize_embeddings=True,
                         show_progress_bar=False, convert_to_numpy=True)[0]
    import numpy as np
    scores = np.asarray(vectors) @ query
    order = np.argsort(-scores)
    selected, total = [], 0
    pages_seen = set()
    for idx in order:
        row = passages[int(idx)]
        page_key = row["url"]
        if page_key in pages_seen: continue
        if total + len(row["text"]) > 3000: continue
        row["score"] = float(scores[idx])
        selected.append(row); pages_seen.add(page_key); total += len(row["text"])
        if len(selected) == 3: break
    return selected


def _internet_query(question):
    query = expand_acronyms(question)
    normalized = _plain(question)
    morocco_terms = ("maroc", "cnss", "cnie", "amo", "idarati", "ramed", "dgi", "anfcc")
    if any(term in normalized for term in morocco_terms):
        if "maroc" not in normalized: query += " Maroc"
    elif not re.search(r"\b(france|maroc|belgique|suisse|canada|espagne|italie|allemagne|tunisie|algerie|royaume uni|etats unis|portugal|pays bas|autriche|suede|norvege|danemark|pologne|irlande|grece|japon|chine|inde|bresil|mexique|australie|nouvelle zelande|egypte|senegal|cameroun|nigeria|cote d ivoire|coree du sud|arabie saoudite|emirats arabes unis|nouvelle caledonie)\b", normalized):
        query += " France"
    return internet_search.sanitize_query(query)


@stage('internet_search')
def _internet_result(question, model):
    query = _internet_query(question)
    pages = internet_search.run_with_timeout(lambda: internet_search.search_pages(query, max_results=6, page_limit=3, timeout=20), timeout=20)
    rows = _rank_web_passages(pages, expand_acronyms(question))
    if not rows:
        raise internet_search.InternetSearchError("Internet est indisponible ou la recherche n'a rien donné.")
    answer_rag.check_model(model)
    result = answer_rag.generate_web(question, rows, model)
    try:
        consulted_date = datetime.now(ZoneInfo("Africa/Casablanca")).date().isoformat()
    except ZoneInfoNotFoundError:
        consulted_date = datetime.now().astimezone().date().isoformat()
    result.update(model=model, source_used="internet", domain="administrative",
                  consulted_at=consulted_date,
                  external_query_sent=any(p.get("external_query_sent", True) for p in pages))
    if result.get("status") != "answered":
        app.logger.warning("Réponse web refusée : %s", result.get("validation", result.get("status")))
        raise internet_search.InternetSearchError("Les pages Internet ont été récupérées, mais le modèle n'a pas produit de réponse avec des citations valides. Reformulez la question.")
    if not _cited_evidence_matches(question, result):
        raise internet_search.InternetSearchError("Internet a trouvé des pages, mais aucune preuve citée ne répond à la question.")
    return result


def _source_used(result):
    used_ids = {e.get("source_id") for claim in result.get("claims", []) for e in claim.get("evidence", [])}
    sources = [s for s in result.get("sources", []) if s.get("source_id") in used_ids]
    kinds = set()
    for source in sources:
        cid = str(source.get("chunk_id", ""))
        if cid.startswith("PDF_"): kinds.add("pdf")
        elif cid.startswith("WEB_"): kinds.add("internet")
        else: kinds.add("service_public")
    if kinds == {"pdf", "service_public"}: return "pdf+service_public"
    if len(kinds) == 1: return next(iter(kinds))
    return "none"


def _cited_evidence_matches(question, result):
    from relevance_guard import relevant
    # A matching source title must not conceal an unrelated quoted passage.
    for claim in result.get('claims', []):
        if any(not relevant(question, proof.get('quote', '')) for proof in claim.get('evidence', [])):
            return False
    quotes = {}
    for claim in result.get("claims", []):
        for evidence in claim.get("evidence", []):
            quotes.setdefault(evidence.get("source_id"), []).append(evidence.get("quote", ""))
    rows = [{"chunk_id": s.get("chunk_id", ""), "title": s.get("title", ""),
             "text": " ".join(quotes.get(s.get("source_id"), []))}
            for s in result.get("sources", []) if s.get("source_id") in quotes]
    return evidence_matches_question(question, rows)


@app.route("/api/chat", methods=["POST"])
@exclusive_operation
def chat():
    return _execute_chat()


def _chat_core():
    data = request.get_json(silent=True) or {}
    started = time.monotonic()
    original_question = (data.get("question") or "").strip()
    session_id = data.get("session_id") or uuid.uuid4().hex
    mode = data.get("mode", "auto")
    if mode == "web": mode = "internet"
    if not original_question:
        return jsonify({"error": "Question vide"}), 400
    history = load_history(session_id)
    if SMALLTALK.match(original_question):
        return _quick_reply(session_id, history + [{"role":"user","content":original_question,"timestamp":datetime.now().isoformat()}],
            "Bonjour ! Posez-moi une question sur une démarche administrative ou sur votre PDF importé.", "smalltalk")
    history.append({"role":"user","content":original_question,"retrieval_question":expand_acronyms(original_question),
                    "doc_id":data.get("doc_id"),"timestamp":datetime.now().isoformat()})
    docs = _session_documents(session_id)
    doc_id = data.get("doc_id")
    if doc_id and doc_id not in [d["doc_id"] for d in docs]:
        raise BadRequest("Ce PDF n'appartient pas à la conversation.")
    named = [d for d in docs if d.get("filename") and _plain(d["filename"]) in _plain(original_question)]
    if named: doc_id = named[-1]["doc_id"]
    is_explain = is_document_overview_request(original_question)
    explicit_pdf = _document_request(original_question) or bool(named) or is_explain
    strategy = "pdf" if mode == "auto" and explicit_pdf else mode
    if strategy == "auto" and docs:
        strategy = "hybrid"
    elif strategy == "auto":
        strategy = "service_public"
    if mode == "auto" and explicit_pdf:
        strategy = "pdf"
    if strategy == "internet":
        domain = is_administrative(original_question)
        if domain == "autre":
            result = {"status":"out_of_domain","domain":"autre","claims":[],"sources":[],"source_used":"none"}
            text = "Je peux vous aider sur les démarches administratives et les documents PDF importés."
        else:
            try:
                model = os.environ.get("RAG_MODEL", answer_rag.MODEL)
                result = _internet_result(original_question, model)
                result["source_notice"] = _source_notice(result)
                text = answer_rag.render(result)
            except Exception as exc:
                result = {"status":"internet_unavailable","domain":"administrative","claims":[],"sources":[],"source_used":"none"}
                text = _internet_failure_message(exc)
        result.update(question=original_question, elapsed_seconds=round(time.monotonic()-started,2))
        history.append({"role":"assistant","content":text,"result":result,"timestamp":datetime.now().isoformat()})
        save_history(session_id,history)
        return jsonify({"session_id":session_id,"answer":text,"result":result})
    try:
        if strategy == "pdf" and not docs:
            result = {"status":"no_pdf","domain":"administrative","claims":[],"sources":[],"source_used":"none"}
            text = "Aucun document PDF n'est joint à cette conversation."
            history.append({"role":"assistant","content":text,"result":result,"timestamp":datetime.now().isoformat()})
            save_history(session_id,history)
            return jsonify({"session_id":session_id,"answer":text,"result":result})
        rows, semantic_top_score, missing = _search_local(
            expand_acronyms(original_question), strategy, docs, doc_id, is_explain,
            relevance_question=original_question)
    except (ValueError, RuntimeError, OSError) as exc:
        if mode == "auto" and not explicit_pdf:
            rows, semantic_top_score, missing = [], None, "local_search_failed"
        else:
            return jsonify({"error":str(exc)}), 500
    domain = "administrative" if docs and is_explain else is_administrative(original_question, semantic_top_score)
    if domain == "autre":
        result={"status":"out_of_domain","domain":"autre","claims":[],"sources":[],"source_used":"none"}
        text="Je peux vous aider sur les démarches administratives et les documents PDF importés."
        history.append({"role":"assistant","content":text,"result":result,"timestamp":datetime.now().isoformat()}); save_history(session_id,history)
        return jsonify({"session_id":session_id,"answer":text,"result":result})
    # Auto cascade: explicite PDF reste privé/local; sinon recherches locales d'abord.
    local_rows = rows
    if local_rows and (mode == "auto" or strategy == "hybrid") and not is_explain and not evidence_matches_question(original_question, local_rows):
        local_rows = []
    model = os.environ.get("RAG_MODEL", answer_rag.MODEL)
    result = None
    if local_rows:
        try:
            if is_explain and strategy == "pdf":
                result = answer_rag.document_overview(local_rows)
            else:
                answer_rag.check_model(model)
                result = answer_rag.generate(expand_acronyms(original_question), local_rows, model)
            result.update(model=model, domain=domain)
            if result.get("status") == "answered":
                if not is_explain and not _cited_evidence_matches(original_question, result):
                    result = None if mode == "auto" else {"status":"insufficient","claims":[],"sources":[],"source_used":"none","domain":domain}
                if result is None:
                    pass
                elif strategy == "internet":
                    result["source_used"]="internet"
                else:
                    result["source_used"]=_source_used(result)
            if result is not None and (mode == "auto" or strategy == "hybrid") and result.get("status") in ("insufficient", "validation_failed"):
                if mode == "auto": result = None
                else:
                    result["source_used"]="none"
                    result["source_notice"]="Aucune source locale pertinente n'a été trouvée."
        except Exception as exc:
            if mode == "auto": result = None
            else: return jsonify({"error":str(exc)}), 500
    else:
        result = None
    should_search_web = mode == "auto" and not explicit_pdf and (result is None or result.get("status") in ("insufficient","validation_failed"))
    if should_search_web:
        try:
            result = _internet_result(original_question, model)
        except Exception as exc:
            result={"status":"internet_unavailable","domain":domain,"claims":[],"sources":[],"source_used":"none"}
            text=_internet_failure_message(exc)
            result.update(question=original_question,elapsed_seconds=round(time.monotonic()-started,2))
            history.append({"role":"assistant","content":text,"result":result,"timestamp":datetime.now().isoformat()}); save_history(session_id,history)
            return jsonify({"session_id":session_id,"answer":text,"result":result})
    if result is None:
        result={"status":"insufficient","domain":domain,"claims":[],"sources":[],"source_used":"none"}
        if strategy == "pdf" or (mode == "auto" and explicit_pdf):
            text="Je n'ai pas trouvé cette information dans le document. Essayez de reformuler la question ou de citer un mot du document."
            result["source_notice"] = "Aucune source pertinente : document PDF uniquement"
        elif strategy in ("service_public","hybrid"):
            text="Je n'ai pas trouvé cette information dans les sources consultées. Essayez de reformuler la question."
            result["source_notice"] = "Aucune source pertinente"
        else:
            text=answer_rag.REFUSAL
    elif result.get("status") != "answered" and (mode != "auto" or explicit_pdf):
        result["source_used"] = "none"
        if strategy == "pdf":
            result["source_notice"] = "Aucune preuve pertinente dans le PDF"
            text = "Je n'ai pas trouvé cette information dans le document. Essayez de reformuler la question ou de citer un mot du document."
        else:
            result["source_notice"] = "Aucune source pertinente. Essayez de reformuler la question."
            text = "Je n'ai pas trouvé cette information dans les sources consultées. Essayez de reformuler la question."
    else:
        result.setdefault("source_used", _source_used(result))
        result["source_notice"]=_source_notice(result)
        text=answer_rag.render(result)
    result.update(question=original_question,retrieval_question=expand_acronyms(original_question),doc_id=doc_id,
                  elapsed_seconds=round(time.monotonic()-started,2))
    history.append({"role":"assistant","content":text,"result":result,"timestamp":datetime.now().isoformat()})
    save_history(session_id,history)
    return jsonify({"session_id":session_id,"answer":text,"result":result})


def _source_notice(result):
    labels={"pdf":"PDF","service_public":"Base Service-Public","pdf+service_public":"PDF + Base Service-Public","internet":"Internet","none":"Aucune source"}
    source=result.get("source_used","none")
    if source == "internet":
        note="Source : Internet"
        if result.get("consulted_at"): note += f" — consulté le {result['consulted_at']}"
        if result.get("external_query_sent"): note += " (la requête nettoyée a été transmise à un moteur externe)"
        return note
    return "Source utilisée : " + labels.get(source, source)


def _internet_failure_message(exc):
    app.logger.error("Échec du mode Internet", exc_info=(type(exc), exc, exc.__traceback__))
    message = str(exc)
    if "n'est pas installé" in message and "RAG_SEARXNG_URL" in message:
        return message
    if message.startswith(("Les pages Internet ont été récupérées", "Des résultats ont été trouvés", "La recherche Internet a dépassé", "Internet a trouvé des pages")):
        return message
    if "Modèle configuré absent" in message or "Ollama" in message:
        return message
    return "Internet est indisponible ou la recherche n'a rien donné. Réessayez plus tard."


def _prewarm():
    """Charge E5 + reranker + LLM au démarrage pour que la 1re question soit rapide."""
    try:
        retrieve_service_public("préchauffage")
    except Exception as exc:
        print(f"Préchauffage de la recherche ignoré : {exc}", flush=True)
    try:
        answer_rag.request_json('/api/generate', {
        'model': answer_rag.MODEL, 'prompt': '', 'think': False, 'keep_alive': answer_rag.KEEP_ALIVE,
            'options': {'num_ctx': answer_rag.NUM_CTX}}, timeout=120)
    except Exception as exc:
        print(f"Préchauffage d'Ollama ignoré : {exc}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    from start_ollama import warn_client_difference
    version = ollama_version()
    if version:
        try: warn_client_difference(version)
        except (OSError, TimeoutError): pass
    threading.Thread(target=_prewarm, daemon=True).start()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()



