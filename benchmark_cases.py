"""Fixed labelled evidence fixtures: measures selection/generation, not retrieval."""
PDF = ('Le loyer mensuel est de 450 euros, charges comprises. '
       'Le dépôt de garantie de 450 euros est versé à la signature. '
       'Le bail est conclu pour une durée de 12 mois. '
       'La date limite de dépôt est le 15 décembre 2026. '
       "Les pièces à fournir sont la CNI, le bail, le RIB, le certificat de scolarité et l'avis d'imposition des parents. "
       'Le numéro du dossier est AL-2026-04817.')
APL = ('Pour demander les APL, remplissez votre demande en ligne sur le site de la Caf et joignez les justificatifs. '
       'Le délai moyen de traitement dépend du régime Caf ou MSA.')
def row(text, cid='PDF_fixture'):
    return {'chunk_id': cid, 'title': 'Dossier administratif', 'url': 'fixture.pdf',
            'chunk_text': text, 'scored_excerpt': text}
CASES = [
    {'question': 'Quel est mon loyer ?', 'rows': [row(PDF)], 'expected': ['450 euros'], 'forbidden': ['signature', '12 mois']},
    {'question': 'Quel est le montant du dépôt de garantie ?', 'rows': [row(PDF)], 'expected': ['450 euros', 'signature'], 'forbidden': ['loyer mensuel']},
    {'question': 'Quelle est la date limite de dépôt ?', 'rows': [row(PDF)], 'expected': ['15 décembre 2026'], 'forbidden': ['signature']},
    {'question': 'Quelles pièces faut-il fournir ?', 'rows': [row(PDF)], 'expected': ['CNI', 'RIB', 'scolarité', 'imposition'], 'forbidden': ['signature']},
    {'question': 'Comment demander les APL ?', 'rows': [row(APL, 'SP_apl')], 'expected': ['en ligne', 'justificatifs'], 'forbidden': ['délai moyen']},
    {'question': "Comment refaire ma carte d'identité ?", 'rows': [row("Pour refaire votre carte d'identité, préparez votre dossier et prenez rendez-vous en mairie.", 'SP_cni')], 'expected': ['rendez-vous', 'mairie'], 'forbidden': []},
    {'question': 'c est quoi la cnss', 'rows': [row(APL, 'SP_apl')], 'expected': [], 'forbidden': [], 'abstain': True},
    {'question': 'recette de couscous', 'rows': [row(PDF)], 'expected': [], 'forbidden': [], 'abstain': True},
    {'question': 'capitale du Japon', 'rows': [row(PDF)], 'expected': [], 'forbidden': [], 'abstain': True},
]
