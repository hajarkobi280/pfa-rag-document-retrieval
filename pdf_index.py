"""Index séparé pour les PDF importés par l'utilisateur (RAG local)."""
import argparse
import io
from local_storage import atomic_write, file_lock
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".runtime"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

PASSAGES_PATH = ROOT / "data" / "pdf_passages.jsonl"
VECTORS_PATH = ROOT / "data" / "pdf_vectors.npy"
REGISTRY_PATH = ROOT / "data" / "pdf_registry.json"


_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import block_pyarrow  # même protection pour le point d’entrée CLI
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(min(4, os.cpu_count() or 2))
    local = ROOT / "models" / "multilingual-e5-small"
    if (local / "download_complete.txt").exists():
        _MODEL = SentenceTransformer(str(local), device="cpu", trust_remote_code=False, local_files_only=True)
    else:
        _MODEL = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu", trust_remote_code=False)
    return _MODEL



def extract_text(pdf_path):
    """Extrait le texte d'un PDF. Essaie plusieurs bibliothèques."""
    pdf_path = Path(pdf_path)
    errors = []
    # pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        if text.strip():
            return text
    except Exception as exc:
        errors.append(f"pypdf: {exc}")
    # PyPDF2
    try:
        from PyPDF2 import PdfReader as R2
        reader = R2(str(pdf_path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        if text.strip():
            return text
    except Exception as exc:
        errors.append(f"PyPDF2: {exc}")
    # pdfminer
    try:
        from pdfminer.high_level import extract_text as pm_extract
        text = pm_extract(str(pdf_path))
        if text.strip():
            return text
    except Exception as exc:
        errors.append(f"pdfminer: {exc}")
    # pypdfium2
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(pdf_path))
        text = "\n".join(page.get_textpage().get_text_range() for page in pdf)
        if text.strip():
            return text
    except Exception as exc:
        errors.append(f"pypdfium2: {exc}")
    raise RuntimeError("Aucun texte exploitable dans ce PDF. Un document scanné nécessite "
                       "une reconnaissance OCR préalable ; un PDF protégé doit être déverrouillé. "
                       "Vérifiez aussi l'installation de pypdf.\n" + "\n".join(errors))


def chunk_text(text, size=1200, overlap=200):
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("Taille et chevauchement invalides")
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _snapshot_path():
    return REGISTRY_PATH.with_suffix(".npz")


def _read_state():
    # Un seul fichier atomique : impossible de publier passages sans vecteurs.
    if _snapshot_path().exists():
        with np.load(_snapshot_path(), allow_pickle=False) as state:
            registry = json.loads(str(state["registry"].item()))
            passages = json.loads(str(state["passages"].item()))
            vectors = state["vectors"].copy()
    else:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8")) if REGISTRY_PATH.exists() else {"documents": []}
        passages = [json.loads(line) for line in PASSAGES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()] if PASSAGES_PATH.exists() else []
        vectors = np.load(VECTORS_PATH, allow_pickle=False) if VECTORS_PATH.exists() else np.empty((0, 384), dtype="float32")
    if vectors.ndim != 2 or len(vectors) != len(passages) or not np.isfinite(vectors).all():
        raise ValueError("Index PDF incohérent : restaurer ou réimporter les documents.")
    if sum(d["chunks"] for d in registry["documents"]) != len(passages):
        raise ValueError("Registre PDF incohérent : restaurer ou réimporter les documents.")
    return registry, passages, vectors


def _commit(registry, passages, vectors):
    if len(passages) != len(vectors) or not np.isfinite(vectors).all():
        raise ValueError("Vecteurs PDF invalides")
    buffer = io.BytesIO()
    np.savez(buffer, registry=json.dumps(registry, ensure_ascii=False),
             passages=json.dumps(passages, ensure_ascii=False), vectors=vectors)
    atomic_write(_snapshot_path(), buffer.getvalue())


def _load_registry():
    return _read_state()[0]


def _load_passages():
    return _read_state()[1]


def ingest(pdf_path, title=None):
    """Extrait, découpe et indexe un PDF. Retourne le doc_id."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(str(pdf_path))
    doc_id = "PDF_" + hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:12]
    text = extract_text(pdf_path)
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("Aucun texte extrait du PDF.")
    # Le calcul peut échouer sans modifier l'index existant.
    with file_lock(REGISTRY_PATH.with_suffix(".lock"), timeout=300):
        registry, passages, old = _read_state()
        existing = next((d for d in registry["documents"] if d["doc_id"] == doc_id), None)
        if existing:
            return doc_id, existing["chunks"], "déjà indexé"
        new_rows = [{"chunk_id": f"{doc_id}_{i}", "doc_id": doc_id,
                     "title": title or pdf_path.name, "url": str(pdf_path),
                     "chunk_text": chunk, "text": chunk, "context": []}
                    for i, chunk in enumerate(chunks)]
        model = _load_model()
        new_vectors = model.encode(["passage: " + r["chunk_text"] for r in new_rows],
                                   normalize_embeddings=True, show_progress_bar=False,
                                   convert_to_numpy=True).astype("float32")
        if new_vectors.ndim != 2 or len(new_vectors) != len(new_rows):
            raise ValueError("Dimensions des embeddings PDF invalides")
        vectors = np.concatenate([old, new_vectors]) if len(old) else new_vectors
        registry["documents"].append({"doc_id": doc_id, "filename": title or pdf_path.name,
                                      "size": pdf_path.stat().st_size,
                                      "path": str(pdf_path), "chunks": len(chunks),
                                      "passage_start": len(passages)})
        _commit(registry, passages + new_rows, vectors)
    return doc_id, len(chunks), "indexé"



def search(question, top_k=5, doc_id=None, min_score=None, doc_ids=None):
    """Recherche dans l'index PDF. Retourne des lignes au format RAG."""
    if top_k < 1 or not question.strip():
        raise ValueError("Question ou limite invalide")
    _, passages, vectors = _read_state()
    allowed = None if doc_ids is None else set(doc_ids)
    if allowed is not None and not allowed:
        return []
    if not passages:
        return []
    model = _load_model()
    query = model.encode(["query: " + question], normalize_embeddings=True,
                        show_progress_bar=False)[0]
    scores = vectors @ query
    results = []
    seen = set()
    for i in np.argsort(-scores):
        row = passages[int(i)]
        if allowed is not None and row["doc_id"] not in allowed:
            continue
        if doc_id and row["doc_id"] != doc_id:
            continue
        if min_score is not None and float(scores[i]) < min_score:
            continue
        if row["chunk_id"] in seen:
            continue
        seen.add(row["chunk_id"])
        results.append({**row, "score": float(scores[i]),
                       "scored_excerpt": row["chunk_text"]})
        if len(results) >= top_k:
            break
    return results


def list_documents():
    return _load_registry().get("documents", [])


def delete_document(doc_id):
    with file_lock(REGISTRY_PATH.with_suffix(".lock")):
        registry, passages, vectors = _read_state()
        docs = registry["documents"]
        if not any(d["doc_id"] == doc_id for d in docs):
            return False
        docs = [d for d in docs if d["doc_id"] != doc_id]
        keep = [i for i, p in enumerate(passages) if p["doc_id"] != doc_id]
        start = 0
        for doc in docs:
            doc["passage_start"] = start
            start += doc["chunks"]
        _commit({"documents": docs}, [passages[i] for i in keep], vectors[keep])
    return True


def clear_all():
    with file_lock(REGISTRY_PATH.with_suffix(".lock")):
        _commit({"documents": []}, [], np.empty((0, 384), dtype="float32"))


def get_document_passages(doc_id=None):
    """Retourne les passages d'un document ou de tous les documents."""
    passages = _load_passages()
    if doc_id:
        return [p for p in passages if p.get("doc_id") == doc_id]
    return passages


def get_document_overview(doc_id=None, max_chunks=4):
    """Retourne les passages initiaux et clés d'un document pour permettre une explication globale."""
    passages = get_document_passages(doc_id)
    if not passages:
        return []
    if max_chunks < 1:
        raise ValueError("Nombre de passages invalide")
    indices = np.linspace(0, len(passages) - 1, min(max_chunks, len(passages)), dtype=int)
    return [{**passages[i], "overview_total_chunks": len(passages),
             "overview_selected_chunks": len(indices)} for i in indices]



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    imp = sub.add_parser("import")
    imp.add_argument("pdf")
    imp.add_argument("--title")
    srch = sub.add_parser("search")
    srch.add_argument("question")
    srch.add_argument("--top-k", type=int, default=5)
    sub.add_parser("list")
    args = parser.parse_args()
    if args.action == "import":
        doc_id, n, status = ingest(args.pdf, args.title)
        print(f"{status} : {doc_id} ({n} passages)")
    elif args.action == "search":
        print(json.dumps(search(args.question, args.top_k), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(list_documents(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


