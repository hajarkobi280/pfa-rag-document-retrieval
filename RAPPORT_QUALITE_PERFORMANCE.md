# Livraison du 26 octobre 2026

Les réponses PDF/base affichent les citations exactes et leurs références. Le texte libre du modèle n'est plus affiché. Le modèle actuel reste `qwen3:1.7b` faute de comparaison complète et satisfaisante. Aucun téléchargement ni changement de version d'Ollama n'a été effectué.

## Changements

- `app.py` : reconnaissance normalisée des demandes d'aperçu ; suppression des messages sur les anciens modes ; métadonnées Ollama ; mesures et journal confidentiel ; route NDJSON `/api/chat/stream` ; conservation de `/api/chat` et du verrou interprocessus. Le flux affiche les étapes, jamais les jetons du modèle.
- `answer_rag.py` : contrôle de présence effective des citations dans les sources ; affichage exclusivement extractif ; tri des preuves par forme de question, recouvrement lexical et score de reclassement du passage ; au plus six unités ; exclusion des unités de délai pour les procédures ; prompt réduit ; génération limitée à 180 jetons. Les scores disponibles sont ceux des passages, pas des phrases : le recouvrement lexical distingue les phrases d'un même passage.
- `rag_observability.py` : durées exclusives par étape, sans compter deux fois les opérations imbriquées ; progression et signal de déconnexion.
- `rerank_search.py`, `internet_search.py` : mesures séparées du reclassement, de la recherche Internet et du téléchargement des pages. Le contexte de mesure traverse les threads Internet.
- `static/app.js` : consommation NDJSON, avancement visible, modèle/version dans l'en-tête et panneau repliable « Détails ». Le panneau « Sources & Preuves vérifiées » est conservé.
- `start_ollama.py` (créé) : priorité au serveur existant, puis au client système ; client `.runtime` seulement en secours ; avertissement de version ; paramètres demandés avant un éventuel lancement masqué.
- `benchmark_models.py`, `benchmark_cases.py` : cas fixes étiquetés, JSON conforme, abstention, preuves attendues/parasites, médiane/p95, mémoire `ollama ps`, variantes revue et chemin rapide, sauvegarde progressive.
- Tests : nouveaux cas qualité, confidentialité, chronométrage, flux, erreurs, déconnexion, cache et préchauffage. Les anciennes assertions exigeant « mode Auto » ont été remplacées.

`block_pyarrow.py` n'a pas été modifié. Les seuils de recherche existants sont conservés. Le singleton `_searcher` est conservé. Le chemin de recherche local utilise désormais le cache existant ; les recherches PDF/base sont effectuées successivement pour obtenir des mesures séparées sans additionner des temps parallèles.

La reformulation optionnelle n'est pas conservée : `PARAPHRASE=False`, y compris si `RAG_PARAPHRASE=1`. Cela évite d'afficher une reformulation incorrecte malgré des nombres présents dans la source.

## Mesures réelles

Machine Windows, Python 3.14, serveur Ollama **0.40.11**, `num_ctx=4096`. Un passage sur neuf questions fixes par variante : six questions avec réponse et trois questions sans réponse dans les preuves fournies. Le benchmark appelle `generate()` directement et **exclut la recherche, le classement de domaine et Internet**.

| Configuration | Médiane (s) | p95 (s) | JSON conforme | Preuves pertinentes, sans parasite | Abstentions correctes |
|---|---:|---:|---:|---:|---:|
| Avant, qwen3:1.7b | 21,232 | 38,342 | 9/9 | 5/6 | 0/3 |
| Après, qwen3:1.7b | 11,249 | 22,189 | 9/9 | 6/6 | 0/3 |
| Après + revue, qwen3:1.7b | 20,791 | 36,909 | 9/9 | 6/6 | 0/3 |
| qwen3:4b | — | — | — | — | absent |
| qwen3.5:4b | — | — | — | — | absent |
| gemma4:e2b | — | — | — | — | absent |

La mémoire rapportée par `ollama ps` est **1,9 Go, CPU 100 %, contexte 4096** pour qwen3:1.7b. Il s'agit de l'empreinte du modèle rapportée par Ollama, pas de toute la mémoire du processus Flask, des index ou de Windows.

Ces mesures sont diagnostiques, pas une garantie de gain : neuf observations seulement, ordre fixe, état de chargement/cache non uniformisé, et tests unitaires ayant tourné pendant certaines mesures. Le p95 utilise le rang supérieur ; avec neuf cas il est égal au maximum. Refaire au moins trois passages sans autre activité pour une comparaison plus stable.

La sélection de preuves corrige le cas APL et les six cas répondables passent. L'abstention de **generate seul** reste mauvaise, même avec revue. Les contrôles applicatifs de domaine et d'adéquation des preuves restent donc nécessaires : les tests de cascade vérifient le refus cuisine/capitale et le repli CNSS vers Internet.

**Décision :** conserver provisoirement qwen3:1.7b ; aucune recommandation comparative définitive. Conserver `RAG_USE_REVIEW=0`, la revue ayant augmenté la latence sans améliorer ces contrôles qualité. Conserver `RAG_FAST_PATH=0` et le cache KV par défaut ; aucun seuil rapide ni gain q8_0 n'a été mesuré.

Gemma 4 E2B existe dans la [bibliothèque officielle Ollama](https://ollama.com/library/gemma4). Sa présence dans le script en fait un candidat à évaluer, pas une garantie qu'il tient confortablement avec les index sur une machine de 8 Go.

Fichiers de résultats : `data/benchmark_before.json`, `data/benchmark_models.json`. Le code de génération avant modification est conservé dans `tmp/answer_rag_before.py` et son lanceur dans `tmp/run_baseline.py`.

## Préchauffage et cache

Vérification réelle dans `data/runtime_validation.json` :

- Préchauffage : **80,947 s**, avec E5 et reclasseur chargés ; Ollama rapporte bien le modèle configuré qwen3:1.7b, contexte 4096.
- Recherche APL après préchauffage : **2,720397 s** au premier appel, **0,0000165 s** au second ; cinq passages, un défaut de cache puis un accès au cache.
- Ce cache concerne la **recherche Service-Public**, pas une réponse complète ni l'appel LLM.
- Le lancement de `start_ollama.py` a reconnu le serveur 0.40.11, n'a lancé aucun autre serveur et a signalé le client du projet 0.40.5.

## Reproduction

Résultat de la suite complète avec `RAG_RUN_SLOW=1` : **71 tests réussis en 104,49 s**, dont la recherche locale réelle et une génération réelle Ollama. Deux avertissements de dépréciation viennent de l'API de quantification PyTorch déjà utilisée par le reclasseur. `node --check static/app.js` passe également.

Dans PowerShell, depuis le dossier du projet :

```powershell
$pythonRag = 'C:\Users\Hajar\AppData\Local\Python\pythoncore-3.14-64\python.exe'
$env:PYTHONPATH = "$PWD\.runtime\test-packages;$PWD\.runtime\python-packages;$PWD\.runtime"
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
& $pythonRag -m pytest -q

# Tests locaux avec Ollama et les index réels
$env:RAG_RUN_SLOW = '1'
& $pythonRag -m pytest -q -m slow
Remove-Item Env:RAG_RUN_SLOW

# Après installation des modèles par vos soins ; aucun téléchargement automatique
& $pythonRag benchmark_models.py --repeats 3 --review

# Comparer uniquement les trois Qwen
& $pythonRag benchmark_models.py --models 'qwen3:1.7b,qwen3:4b,qwen3.5:4b' --repeats 3 --review

# Vérification réelle du cache et du préchauffage
& $pythonRag tmp/validate_runtime.py
```

`--cases chemin.json` permet d'utiliser d'autres preuves annotées, notamment de vrais passages retrouvés avec `rerank_score`. Chaque cas contient `question`, `rows`, `expected` (sous-chaînes attendues), `forbidden` (parasites) et éventuellement `abstain: true`.

Pour tester un chemin rapide, fournir **un seuil préalablement calibré** avec `--fast-threshold VALEUR --cases chemin.json`. Les données fixes livrées n'inventent aucun score de reclasseur et ne permettent donc pas de mesurer des activations rapides. La variante rapide doit être comparée au standard sur le même jeu ; `fast_path_hits` indique les appels réellement évités.

`--max-median-seconds VALEUR` permet de préciser votre budget de latence. `--save-default` n'enregistre `data/model_recommendation.json` que si tous les candidats demandés sont mesurés et qu'un modèle réussit tous les contrôles qualité étiquetés. `RAG_MODEL` reste prioritaire ; sans recommandation mesurée, la valeur de secours reste qwen3:1.7b.

Redémarrer **Flask** pour charger les modifications. Il n'est pas nécessaire de redémarrer le serveur Ollama existant. Les nouveaux réglages de serveur ne peuvent prendre effet que lors d'un futur lancement effectué par `start_ollama.py`.

## Durées et confidentialité

`result.timings` contient `classification`, `pdf_search`, `base_search`, `reranking`, `llm_generation`, `review`, `internet_search`, `page_download`, `total`, en secondes. Une étape non exécutée vaut zéro. Les durées des sous-étapes sont retirées des étapes englobantes. Le total inclut aussi le travail non attribué à ces étapes, par exemple les appels de statut.

`data/timings.jsonl` contient les durées et métadonnées, sans question ni citations par défaut. `RAG_LOG_QUESTIONS=1` ajoute explicitement la question à ce journal. L'historique conversationnel existant garde son fonctionnement normal.

## Limites et atténuations

- Une citation exacte peut être hors sujet. Le tri, les contrôles applicatifs et les tests étiquetés réduisent le risque ; le petit modèle ne sait toujours pas s'abstenir de façon fiable sur ce jeu. Comparer les modèles 4B avant de modifier le défaut.
- Une segmentation par phrase peut séparer un intitulé d'une liste ou une condition d'une exception. Élargir les cas de validation à des PDF longs, tableaux et listes avant de généraliser les résultats.
- Le chemin rapide exige un score de **reclassement**, un seuil explicite, des termes de question présents et au plus deux unités restantes. Il reste désactivé et non recommandé tant qu'un benchmark sur de vrais scores ne montre pas la qualité attendue.
- Le contrôle de présence des citations ne prouve pas que la source Internet est correcte ou récente. Le panneau conserve les URL, le statut officiel et la date de consultation ; privilégier les pages officielles.
- À la déconnexion, les étapes suivantes sont annulées et le verrou est libéré à la sortie du travail. Un appel bloquant déjà envoyé à Ollama peut continuer jusqu'à sa réponse ou son délai maximal ; le verrou reste alors tenu pour éviter deux traitements concurrents.
- Les paramètres q8_0 et le comportement des modèles absents ne sont pas validés. Aucun résultat n'est déduit de leur nom ou de leur taille.
