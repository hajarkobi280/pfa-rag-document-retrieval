"""Measure les scores RAG locaux et propose seulement des seuils séparables par les données."""
import argparse, csv, json, os
from pathlib import Path

QUESTIONS = [
 ("Comment demander les APL ?", "administrative", True),
 ("Comment faire une demande de logement social HLM ?", "administrative", True),
 ("Comment refaire ma carte d'identité ?", "administrative", True),
 ("Que faire en cas de perte de passeport ?", "administrative", True),
 ("Comment s'inscrire sur les listes électorales ?", "administrative", True),
 ("c est quoi la cnss", "administrative", False),
 ("comment obtenir la CNIE au Maroc", "administrative", False),
 ("comment créer une SARL au Maroc", "administrative", False),
 ("comment renouveler le permis de conduire au Maroc", "administrative", False),
 ("c est quoi l'AMO", "administrative", False),
 ("recette de couscous", "autre", False),
 ("quelle est la capitale du japon", "autre", False),
 ("écris-moi un poème sur la mer", "autre", False),
 ("qui a gagné la coupe du monde 2018", "autre", False),
]

def separating_midpoint(positive, negative):
    positive=[float(x) for x in positive if x is not None]
    negative=[float(x) for x in negative if x is not None]
    if not positive or not negative or min(positive) <= max(negative): return None
    return (min(positive)+max(negative))/2

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions',type=Path,help='CSV question,domain,in_base avec valeurs administrative/autre et yes/no')
    parser.add_argument('--pdf-question',action='append',default=[],help='Question positive dans le PDF indexé')
    parser.add_argument('--pdf-csv',type=Path,help='CSV question,relevant (yes/no) pour exemples PDF positifs et négatifs')
    parser.add_argument('--output',type=Path,default=Path('data/calibration_thresholds.json'))
    args=parser.parse_args()
    records=QUESTIONS
    if args.questions:
        with args.questions.open(encoding='utf-8-sig',newline='') as f:
            records=[(r['question'],r['domain'],r['in_base'].strip().lower() in ('yes','y','true','1','oui')) for r in csv.DictReader(f)]
    # Ignore runtime thresholds so measurements are not clipped by the values being calibrated.
    os.environ.pop("RAG_SEM_MIN", None)
    os.environ.pop("RAG_MIN_SCORE", None)
    from hybrid_search import HybridSearch
    from rerank_search import RerankedSearch
    import pdf_index
    from app import is_administrative
    engine=HybridSearch(); reranker=RerankedSearch()
    pdf_docs=pdf_index.list_documents()
    rows=[]
    for q,expected_domain,in_base in records:
        hybrid=engine.search(q,top_k=5,candidates=40,include_candidates=False)
        ranked=reranker.search(q,top_k=1,pool_size=12)
        domain=is_administrative(q,hybrid.get('semantic_top_score'))
        pdf_top=pdf_index.search(q,top_k=1) if pdf_docs else []
        rows.append({'question':q,'expected_domain':expected_domain,'expected_in_base':in_base,
            'domain_detected':domain,'semantic_top_score':hybrid.get('semantic_top_score'),
            'lexical_hits':hybrid.get('lexical_hits'),
            'rerank_score_top1':(ranked.get('reranked') or [{}])[0].get('rerank_score'),
            'pdf_score_top1':float(pdf_top[0]['score']) if pdf_top else None})
        print('Mesuré : '+q,flush=True)
    pdf_measure=[]; pdf_relevant=[]; pdf_irrelevant=[]
    pdf_queries=[(q,True) for q in args.pdf_question]
    if args.pdf_csv:
        with args.pdf_csv.open(encoding='utf-8-sig',newline='') as f:
            pdf_queries += [(r['question'],r['relevant'].strip().lower() in ('yes','y','true','1','oui')) for r in csv.DictReader(f)]
    for q,relevant in pdf_queries:
        found=pdf_index.search(q,top_k=1)
        score=float(found[0]['score']) if found else None
        pdf_measure.append({'question':q,'relevant':relevant,'score_top1':score})
        if score is not None: (pdf_relevant if relevant else pdf_irrelevant).append(score)
    positives=[r for r in rows if r['expected_in_base']]
    negatives=[r for r in rows if not r['expected_in_base']]
    sem_min=separating_midpoint([r['semantic_top_score'] for r in positives],[r['semantic_top_score'] for r in negatives])
    min_score=separating_midpoint([r['rerank_score_top1'] for r in positives],[r['rerank_score_top1'] for r in negatives])
    pdf_hybrid=separating_midpoint(pdf_relevant,pdf_irrelevant)
    all_pdf_scores=pdf_relevant+pdf_irrelevant
    pdf_only=min(all_pdf_scores) if pdf_hybrid is not None and all_pdf_scores else None
    proposals={'SEM_MIN':sem_min,'MIN_SCORE':min_score,'RAG_PDF_MIN_HYBRID':pdf_hybrid,
        'RAG_PDF_MIN_PDFONLY':pdf_only,
        'note':'null signifie données insuffisantes ou classes qui se chevauchent. PDFONLY est le minimum mesuré des scores annotés et reste volontairement inférieur au seuil hybride.'}
    report={'measurements':rows,'pdf_measurements':pdf_measure,'proposals':proposals}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('question | domaine détecté | base attendue | semantic_top_score | lexical_hits | rerank top1 | PDF top1')
    for row in rows:
        print(f"{row['question']} | {row['domain_detected']} | {'oui' if row['expected_in_base'] else 'non'} | {row['semantic_top_score']} | {row['lexical_hits']} | {row['rerank_score_top1']} | {row['pdf_score_top1']}")
    print('Seuils proposés : '+json.dumps(proposals,ensure_ascii=False))
    print('Rapport détaillé : '+str(args.output))
if __name__=='__main__': main()
