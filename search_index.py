"""Index lexical local TF-IDF, sans téléchargement de modèle."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/service_public_clean.jsonl"
INDEX = ROOT / "data/index_lexical"
STOP_WORDS = "a au aux avec ce ces cette cet dans de des du elle en et eux il ils je la le les leur lui ma mais me mes moi mon ne nos notre nous on ou par pas pour qu que quel quelle quelles quels qui sa se ses son sur ta te tes toi ton tu un une vos votre vous comment puis peux peut faut faire est sont etre ai as avons avez ont suis es sommes etes quand dont si cela ca".split()


def build():
    INDEX.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in SOURCE.open(encoding="utf-8")]
    vectorizer = TfidfVectorizer(strip_accents="unicode", lowercase=True,
        ngram_range=(1, 2), sublinear_tf=True, max_df=0.95,
        max_features=150000, dtype=np.float32, stop_words=STOP_WORDS)
    vectors = vectorizer.fit_transform(row["title"] + "\n" + row["title"] + "\n" + row["chunk_text"] for row in rows)
    save_npz(INDEX / "vectors.npz", vectors)
    joblib.dump(vectorizer, INDEX / "vectorizer.joblib")
    with (INDEX / "passages.jsonl").open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps({k: row[k] for k in ("chunk_id", "doc_id", "title", "url", "chunk_text")}, ensure_ascii=False) + "\n")
    manifest = {"method": "TF-IDF lexical, unigrammes et bigrammes", "passages": len(rows),
        "features": vectors.shape[1], "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
    (INDEX / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def search(question, top_k=5, max_per_document=1):
    if not question.strip():
        raise ValueError("La question est vide.")
    # Charger uniquement le fichier joblib local créé par ce projet.
    vectorizer = joblib.load(INDEX / "vectorizer.joblib")
    vectors = load_npz(INDEX / "vectors.npz")
    query = vectorizer.transform([question])
    scores = (vectors @ query.T).toarray().ravel()
    rows = [json.loads(line) for line in (INDEX / "passages.jsonl").open(encoding="utf-8")]
    if top_k < 1 or max_per_document < 1:
        raise ValueError("Les limites doivent être positives.")
    result, counts = [], {}
    for i in np.argsort(-scores):
        if scores[i] <= 0:
            break
        row = rows[int(i)]
        document = row["doc_id"]
        if counts.get(document, 0) >= max_per_document:
            continue
        counts[document] = counts.get(document, 0) + 1
        result.append({**row, "score": float(scores[i])})
        if len(result) == top_k:
            break
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("build")
    finder = sub.add_parser("search")
    finder.add_argument("question")
    finder.add_argument("--top-k", type=int, default=5)
    finder.add_argument("--max-per-document", type=int, default=1)
    args = parser.parse_args()
    if args.action == "build":
        build()
    else:
        if args.top_k < 1 or args.max_per_document < 1:
            parser.error("Les limites doivent être positives")
        print(json.dumps(search(args.question, args.top_k, args.max_per_document), ensure_ascii=False, indent=2))
