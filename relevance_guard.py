"""Conservative lexical topic guard, complementary to semantic retrieval.

This detects missing subjects, not entailment. Generic questions pass through.
Threshold calibration uses annotated fixtures, never guessed reranker logits.
"""
import json
import re
import unicodedata
from pathlib import Path

STOP = set('a au aux avec ce ces cette cet comment de des du dans d en est et faire faut fournir il je la le les ma mes moi mon nous on ou par pour pourquoi qu que quel quelle quelles quels qui se son sur un une vos votre vous c quoi cout montant prix somme demander demande obtenir refaire renouveler pieces justificatifs dossier document pdf information informations contient explique resume'.split())
ALIASES = {'mensuel': 'mensualite', 'mensuelle': 'mensualite', 'delais': 'delai',
           'cni': 'identite', 'apl': 'apl'}

def tokens(text):
    value = ''.join(c for c in unicodedata.normalize('NFD', text.lower()) if unicodedata.category(c) != 'Mn')
    words = re.findall(r'[a-z0-9]+', value)
    return {ALIASES.get(w, w) for w in words if len(w) > 1 and w not in STOP}

def topic_score(question, quote):
    subject = tokens(question)
    if not subject:
        return None  # absence of a lexical subject is not evidence of irrelevance
    return len(subject & tokens(quote)) / len(subject)

def calibrated_threshold():
    try:
        data = json.loads((Path(__file__).resolve().parent/'data/topic_calibration.json').read_text(encoding='utf-8'))
        if data.get('validated') and data.get('feature_version') == 1:
            return float(data['threshold'])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return 0.0  # only reject complete absence of shared subject terms

def relevant(question, quote):
    score = topic_score(question, quote)
    return score is None or score > calibrated_threshold()
