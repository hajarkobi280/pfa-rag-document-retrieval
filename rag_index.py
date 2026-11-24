"""Index vectoriel local : py rag_index.py build / search \"question\"."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".runtime"))
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import block_pyarrow  # environnement local : éviter le chargement de arrow.dll
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL = "intfloat/multilingual-e5-small"
SOURCE = ROOT / "data" / "service_public_clean.jsonl"
INDEX = ROOT / "data" / "index_service_public"


def save_json(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def load_model():
    torch.set_num_threads(min(4, os.cpu_count() or 2))
    print("Chargement du modèle multilingue...", flush=True)
    local = ROOT / "models/multilingual-e5-small"
    if (local / "download_complete.txt").exists():
        return SentenceTransformer(str(local), device="cpu", trust_remote_code=False, local_files_only=True)
    return SentenceTransformer(MODEL, device="cpu", trust_remote_code=False)


def build(batch_size):
    INDEX.mkdir(parents=True, exist_ok=True)
    checksum = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    state_path = INDEX / "progress.json"
    model = load_model()
    config = {"source_sha256": checksum, "model": MODEL, "window_tokens": 440, "overlap_tokens": 60, "version": 1}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["config"] != config:
            raise SystemExit("La source ou la configuration a changé. Utiliser un autre dossier d'index.")
        rows = [json.loads(line) for line in (INDEX / "passages.jsonl").open(encoding="utf-8")]
    else:
        print("Préparation des passages, avec découpage des textes longs...", flush=True)
        rows = []
        originals = 0
        with SOURCE.open(encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                originals += 1
                text = row["chunk_text"]
                offsets = model.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)["offset_mapping"]
                for start in range(0, len(offsets), 380):
                    end = min(start + 440, len(offsets))
                    fragment = text[offsets[start][0]:offsets[end-1][1]]
                    rows.append({"chunk_id": row["chunk_id"], "doc_id": row["doc_id"], "title": row["title"], "url": row["url"], "audience": row.get("audience"), "theme": row.get("theme"), "part": start // 380, "text": fragment})
                    if end == len(offsets):
                        break
        temporary = INDEX / "passages.jsonl.part"
        with temporary.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(INDEX / "passages.jsonl")
        state = {"config": config, "original_passages": originals, "vectors": len(rows), "completed": 0, "dimension": model.get_sentence_embedding_dimension()}
        array = np.lib.format.open_memmap(INDEX / "vectors.npy", mode="w+", dtype="float32", shape=(len(rows), state["dimension"]))
        array.flush()
        del array
        save_json(state_path, state)
    vectors = np.load(INDEX / "vectors.npy", mmap_mode="r+")
    total = len(rows)
    for start in range(state["completed"], total, batch_size):
        end = min(start + batch_size, total)
        texts = ["passage: " + row["text"] for row in rows[start:end]]
        vectors[start:end] = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        vectors.flush()
        state["completed"] = end
        save_json(state_path, state)
        if start == 0 or end % (batch_size * 10) == 0 or end == total:
            print(f"Vecteurs calculés : {end}/{total} ({100*end/total:.1f} %)", flush=True)
    save_json(INDEX / "manifest.json", {**state, "status": "complete", "metric": "cosine", "query_prefix": "query: ", "passage_prefix": "passage: "})
    print(f"Index terminé : {total} vecteurs. Dossier : {INDEX}", flush=True)


def search(question, top_k, max_per_document=1):
    if not question.strip():
        raise SystemExit("La question ne doit pas être vide.")
    if not (INDEX / "manifest.json").exists():
        raise SystemExit("Index incomplet. Exécuter d'abord : py rag_index.py build")
    manifest = json.loads((INDEX / "manifest.json").read_text(encoding="utf-8"))
    vectors = np.load(INDEX / "vectors.npy", mmap_mode="r")
    if len(vectors) != manifest["completed"]:
        raise SystemExit("Index incohérent.")
    model = load_model()
    query = model.encode(["query: " + question], normalize_embeddings=True, show_progress_bar=False)[0]
    scores = vectors @ query
    rows = [json.loads(line) for line in (INDEX / "passages.jsonl").open(encoding="utf-8")]
    selected, seen, counts = [], set(), {}
    for i in np.argsort(-scores):
        row = rows[int(i)]
        if row["chunk_id"] in seen:
            continue
        if counts.get(row["doc_id"], 0) >= max_per_document:
            continue
        seen.add(row["chunk_id"])
        counts[row["doc_id"]] = counts.get(row["doc_id"], 0) + 1
        selected.append({**row, "score": float(scores[i])})
        if len(selected) == top_k:
            break
    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    builder = sub.add_parser("build")
    builder.add_argument("--batch-size", type=int, default=16)
    finder = sub.add_parser("search")
    finder.add_argument("question")
    finder.add_argument("--top-k", type=int, default=5)
    finder.add_argument("--max-per-document", type=int, default=1)
    args = parser.parse_args()
    if args.action == "build":
        if args.batch_size < 1:
            parser.error("batch-size doit être positif")
        build(args.batch_size)
    else:
        if args.top_k < 1 or args.max_per_document < 1:
            parser.error("Les limites doivent être positives")
        search(args.question, args.top_k, args.max_per_document)
