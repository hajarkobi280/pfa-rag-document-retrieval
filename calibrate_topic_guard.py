"""Quick offline calibration and independent holdout; no LLM/network required."""
import json
from pathlib import Path
from relevance_guard import topic_score
from benchmark_cases import CASES, PDF, APL

# Fit on the original fixed benchmark; evaluate on new paired examples.
TRAIN = [(c['question'], ' '.join(r['chunk_text'] for r in c['rows']), not c.get('abstain', False)) for c in CASES]
HOLDOUT = [
    ('Combien coûte mon loyer ?', 'Le loyer est de 520 euros.', True),
    ('Quelle est la caution ?', 'La caution est fixée à 500 euros.', True),
    ('Quelle est la durée du bail ?', 'Le bail dure douze mois.', True),
    ('Quand expire mon passeport ?', 'Votre passeport expire en juin 2027.', True),
    ('Comment obtenir une CNI ?', "La carte d'identité se demande en mairie.", True),
    ('Comment déposer une demande APL ?', APL, True),
    ('Quelles pièces fournir ?', 'Une CNI et un RIB sont nécessaires.', True),
    ('Quel est le numéro du dossier ?', 'Le numéro du dossier est AL-2026-04817.', True),
    ('Quel est le montant de mon salaire ?', PDF, False),
    ('Quelle est la date du mariage ?', 'La date limite de dépôt est en décembre.', False),
    ('Comment demander un passeport ?', APL, False),
    ('Comment contacter la CNSS ?', APL, False),
    ('Quel est le montant des impôts ?', 'Le montant du loyer est de 450 euros.', False),
    ('Comment obtenir un visa ?', "La carte d'identité se demande en mairie.", False),
    ('Quelle est la capitale du Canada ?', PDF, False),
    ('Comment préparer un gâteau ?', APL, False),
]

def evaluate(cases, threshold):
    rows = []
    for question, quote, expected in cases:
        score = topic_score(question, quote)
        accepted = score is None or score > threshold
        rows.append(dict(question=question, score=score, expected=expected, accepted=accepted))
    return {'positive_cases': sum(r['expected'] for r in rows),
            'negative_cases': sum(not r['expected'] for r in rows),
            'false_refusals': sum(r['expected'] and not r['accepted'] for r in rows),
            'false_acceptances': sum(not r['expected'] and r['accepted'] for r in rows),
            'measurements': rows}

def main():
    scores = {topic_score(q, t) for q, t, _ in TRAIN} - {None}
    candidates = [(t, evaluate(TRAIN, t)) for t in sorted(scores)]
    safe = [(t,r) for t,r in candidates if r['false_refusals'] == 0]
    threshold, train = min(safe, key=lambda item:(item[1]['false_acceptances'],item[0]))
    holdout = evaluate(HOLDOUT, threshold)
    report = {'feature_version': 1, 'threshold': threshold,
              'validated': holdout['false_refusals'] == 0 and holdout['false_acceptances'] == 0,
              'scope': 'lexical subject coverage, not semantic or reranker calibration',
              'train': train, 'holdout': holdout}
    path = Path(__file__).resolve().parent/'data/topic_calibration.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
