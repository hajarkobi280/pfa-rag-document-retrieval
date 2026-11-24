# Contrôle rapide de pertinence — 27 octobre 2026

Un filtre lexical de sujet est ajouté avant la sélection par le LLM. Les termes génériques (« montant », « demande », « document »…) ne suffisent plus à justifier un passage. Si aucune unité ne partage un terme de sujet, la génération s'abstient sans appel Ollama. Les questions génériques sans sujet identifiable restent admises pour éviter un refus systématique des listes de pièces. Une équivalence CNI/identité est prise en charge.

Le contrôle applicatif des preuves citées vérifie aussi leur texte : un titre de source pertinent ne peut plus, à lui seul, justifier une citation sans rapport.

## Calibration rapide

Script : `calibrate_topic_guard.py`. Mesures : `data/topic_calibration.json`.

- Ajustement sur les 9 questions fixes existantes : 6 positives et 3 négatives.
- Validation distincte sur 16 nouveaux couples question/passage : 8 positifs et 8 négatifs.
- Seuil proposé par les scores observés : recouvrement du sujet strictement supérieur à zéro.
- Ajustement : 0 faux refus, 0 fausse acceptation.
- Validation : 0 faux refus sur 8 positifs, 1 fausse acceptation sur 8 négatifs.

Le fichier marque donc `validated: false` : aucun seuil présenté comme universellement validé n'est activé. Le filtre conserve sa règle minimale explicite d'absence totale de termes de sujet. Les seuils sémantiques E5 et les logits du reclasseur ne sont pas modifiés. Il ne s'agit pas d'une calibration sémantique complète.

Limite laissée visible : « Quelle est la date du mariage ? » partage « date » avec une date limite de dépôt. Le filtre ne prouve donc pas que la citation répond à la question. Il peut aussi manquer des synonymes : une validation plus large est nécessaire avant d'en tirer une conclusion générale. Le jeu de validation n'a pas été retouché pour masquer son erreur.

## Comparaison des modèles

Le serveur ne dispose que de `qwen3:1.7b`. Aucun téléchargement n'a été lancé. La comparaison 4B reste impossible tant que ces modèles ne sont pas installés.

Une passe du benchmark existant avec le nouveau filtre est enregistrée séparément dans `data/benchmark_topic_guard.json`. Les résultats historiques et les figures du rapport précédent ne sont pas écrasés. Les abstentions sans appel LLM sont exclues du taux de JSON valide et ne constituent pas une amélioration du modèle lui-même.

## Reproduction

```powershell
$pythonRag = 'C:\Users\Hajar\AppData\Local\Python\pythoncore-3.14-64\python.exe'
& $pythonRag calibrate_topic_guard.py
& $pythonRag benchmark_models.py --models qwen3:1.7b --repeats 1 --output data/benchmark_topic_guard.json
```

Les nouveaux tests couvrent les abstentions sans génération, les sujets pertinents et non pertinents, les questions génériques, les titres trompeurs et la limite observée de la calibration.

Résultats : **90 tests réussis, 2 tests lents non exécutés**. Le benchmark avec Ollama réel conserve **6/6 cas répondables corrects** et obtient **3/3 abstentions correctes**, grâce au filtre avant génération ; seuls les six cas répondables appellent le modèle. JSON conforme : 6/6 appels. Ce résultat sur neuf cas ne constitue pas une comparaison avec les modèles 4B ni une précision générale.
