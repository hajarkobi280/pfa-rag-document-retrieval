"""Recherche hybride locale, fusion des rangs (RRF), sans reconstruire les index."""
import argparse
import json
from pathlib import Path

from rag_index import load_model, INDEX as SEMANTIC_INDEX
import joblib
import numpy as np
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parent
LEXICAL_INDEX = ROOT / "data/index_lexical"


def fuse(lexical, semantic, constant=60):
    """Chaque liste contient des chunk_id uniques, dans l'ordre de pertinence."""
    fused = {}
    for name, ranking in (("lexical", lexical), ("semantic", semantic)):
        for rank, chunk_id in enumerate(ranking, 1):
            item = fused.setdefault(chunk_id, {"score": 0.0, "ranks": {}})
            item["score"] += 1.0 / (constant + rank)
            item["ranks"][name] = rank
    return sorted(fused.items(), key=lambda pair: (-pair[1]["score"], pair[0]))


def top_indices(scores, n):
    """Les n meilleurs indices, triés (tri partiel : bien plus rapide qu'un tri complet)."""
    n = max(1, min(n, len(scores)))
    if n >= len(scores):
        return np.argsort(-scores, kind="stable")
    idx = np.argpartition(-scores, n - 1)[:n]
    return idx[np.argsort(-scores[idx], kind="stable")]


class HybridSearch:
    def __init__(self):
        lex_manifest = json.loads((LEXICAL_INDEX / "manifest.json").read_text(encoding="utf-8"))
        sem_manifest = json.loads((SEMANTIC_INDEX / "manifest.json").read_text(encoding="utf-8"))
        if sem_manifest.get("status") != "complete":
            raise ValueError("Index sémantique incomplet")
        if lex_manifest["source_sha256"] != sem_manifest["config"]["source_sha256"]:
            raise ValueError("Les deux index ne proviennent pas de la même base")
        self.lex = load_npz(LEXICAL_INDEX / "vectors.npz")
        self.vectorizer = joblib.load(LEXICAL_INDEX / "vectorizer.joblib")
        self.rows = [json.loads(line) for line in (LEXICAL_INDEX / "passages.jsonl").open(encoding="utf-8")]
        self.by_id = {row["chunk_id"]: row for row in self.rows}
        # Chargé en RAM (au lieu de mmap) : pas de relecture disque à chaque question.
        self.sem = np.ascontiguousarray(np.load(SEMANTIC_INDEX / "vectors.npy"), dtype=np.float32)
        self.windows = [json.loads(line) for line in (SEMANTIC_INDEX / "passages.jsonl").open(encoding="utf-8")]
        if len(self.sem) != sem_manifest["completed"] or len(self.sem) != len(self.windows) or self.lex.shape[0] != len(self.rows):
            raise ValueError("Dimensions des index incohérentes")
        if not all(row["chunk_id"] in self.by_id for row in self.windows):
            raise ValueError("Passages sémantiques sans correspondance lexicale")
        self.model = load_model()

    def search(self, question, top_k=5, candidates=100, max_per_document=1, include_candidates=False):
        if not question.strip() or min(top_k, candidates, max_per_document) < 1:
            raise ValueError("Question vide ou limites invalides")
        if candidates < top_k:
            raise ValueError("candidates doit être au moins égal à top-k")
        lexical_scores = (self.lex @ self.vectorizer.transform([question]).T).toarray().ravel()
        lex_idx = top_indices(lexical_scores, candidates)
        lexical = [self.rows[int(i)]["chunk_id"] for i in lex_idx if lexical_scores[i] > 0]
        query = self.model.encode(["query: " + question], normalize_embeddings=True, show_progress_bar=False)[0]
        semantic_scores = self.sem @ query
        semantic, seen = [], set()
        # Une seule contribution par passage original, même s'il a plusieurs fenêtres.
        # candidates * 4 : marge pour les chunks qui ont plusieurs fenêtres.
        sem_idx = top_indices(semantic_scores, candidates * 4)
        for i in sem_idx:
            chunk_id = self.windows[int(i)]["chunk_id"]
            if chunk_id not in seen:
                seen.add(chunk_id)
                semantic.append(chunk_id)
                if len(semantic) == candidates:
                    break
        merged = fuse(lexical, semantic)

        def select(ranking):
            results, counts = [], {}
            for chunk_id, info in ranking:
                row = self.by_id[chunk_id]
                document = row["doc_id"]
                if counts.get(document, 0) >= max_per_document:
                    continue
                counts[document] = counts.get(document, 0) + 1
                results.append({**row, **info})
                if len(results) == top_k:
                    break
            return results

        result = {
            "question": question,
            "lexical": select([(cid, {"rank": rank}) for rank, cid in enumerate(lexical, 1)]),
            "semantic": select([(cid, {"rank": rank}) for rank, cid in enumerate(semantic, 1)]),
            "hybrid": select(merged),
            # Signaux « hors base » utilisés par rerank_search.py
            "semantic_top_score": float(semantic_scores[sem_idx[0]]),
            "lexical_hits": len(lexical),
        }
        if include_candidates:
            result['candidate_pool'] = [{**self.by_id[cid], **info} for cid, info in merged]
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--max-per-document", type=int, default=1)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if not args.smoke_test and not args.question:
        parser.error("Indiquer une question")
    if min(args.top_k, args.candidates, args.max_per_document) < 1 or args.candidates < args.top_k:
        parser.error("Limites positives requises, candidates >= top-k")
    engine = HybridSearch()
    if args.smoke_test:
        cases = [
            ("Comment demander une aide au logement ?", "F12006"),
            ("Comment faire une demande de logement social HLM ?", "F10007"),
            ("Les cotisations syndicales donnent-elles un crédit d'impôt ?", "F1"),
            ("Un étudiant peut-il obtenir une aide pour payer son loyer ?", "F1563"),
            ("Quelles conditions pour obtenir Mobili-Jeune en alternance ?", "F33173"),
        ]
        report = {"note": "Tests exploratoires sur 5 questions choisies ; références documentaires indicatives, pas un benchmark validé ni une validation des réponses.", "results": []}
        for question, expected in cases:
            result = engine.search(question)
            summary = {"question": question, "expected_doc_id": expected}
            for method in ("lexical", "semantic", "hybrid"):
                rows = result[method]
                assert len({r["doc_id"] for r in rows}) == len(rows)
                summary[method] = {"expected_in_top5": expected in [r["doc_id"] for r in rows], "results": [{k: r[k] for k in ("doc_id", "chunk_id", "title", "url")} for r in rows]}
            report["results"].append(summary)
            print(f"Test terminé : {question}", flush=True)
        path = ROOT / "data/comparaison_recherche.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Rapport : {path}")
    else:
        results = engine.search(args.question, args.top_k, args.candidates, args.max_per_document)
        print(json.dumps(results if args.compare else results["hybrid"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()