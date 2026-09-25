# Édition factuelle naturelle — journal exploratoire

**25 septembre 2026 — premier jalon obtenu.** À la demande explicite de l’utilisateur, cette phase autorise les ajustements de prompts, positions et paramètres. Les résultats sont exploratoires. Les expériences confirmatoires précédentes restent closes.

Objectif de cette session : obtenir volontairement « He » au lieu de « H » comme symbole de l’hydrogène dans une réponse naturelle, puis regarder quelques questions voisines. Budget commun : 128 forwards ou 60 minutes, un seul modèle chargé, MemoryHigh=22G, MemoryMax=24G, MemorySwapMax=8G.

## Essai et résultat

1. Réponse native complète : `The chemical symbol of hydrogen is **H**.`
2. Capture d’un contexte donneur sur l’hélium, avec le préambule de réponse fourni pour apprendre la direction. Ce donneur prédit `He`.
3. Premier essai : ajouter **résidu donneur − résidu receveur**, après L31, gain 1, à la position précédant le huitième token de réponse. Le modèle choisit `He`. Les alternatives gain 2 et L41 n’ont pas été exécutées.
4. Poursuite sans contrainte de tokens : `The chemical symbol of hydrogen is **He**.` Le modèle produit lui-même la ponctuation et le token de fin de tour.
5. Même vecteur, même gain et même position relative sur trois questions voisines :

| Question / formulation | Avant | Après |
|---|---|---|
| `The chemical symbol of hydrogen is:` | `H` | **`He`** |
| `The chemical symbol of helium is:` | `He` | `He` |
| `The atomic number of hydrogen is:` | `1` | `1` |
| `What is the chemical symbol for hydrogen?` | `H` | **`He`** |

Les huit réponses avant/après sont complètes et se terminent naturellement. Pour économiser le calcul, les sept tokens de préambule déjà générés naturellement sont réutilisés avant l’intervention ; celui du numéro atomique provient de l’audit précédent. Tous les nouveaux tokens sont des argmax libres sur le vocabulaire complet. Aucun masque, remplacement de token ou changement du texte produit.

## Ce que cela montre et ce qui reste à faire

Le changement factuel observable fonctionne sur la formulation initiale et une reformulation. Les deux autres réponses visibles restent identiques. Leurs scores internes changent cependant : pour le numéro atomique, la marge du premier token sur son concurrent passe d’environ 12,99 à 5,97. Cela ne démontre donc pas l’absence d’effets collatéraux.

Le vecteur ajouté vaut environ 19,2 % de la norme du résidu receveur. Sur la formulation initiale, le logit de `He` passe de 6,77 à 26,55, tandis que celui de `H` passe de 26,81 à 19,08.

C’est une **intervention temporaire sur les activations**, à une position choisie manuellement. Les poids de base sont inchangés. Le même déplacement est appliqué aux quatre questions, sans détecteur sémantique. À chaque recalcul du préfixe, il est réappliqué à sa position absolue d’origine, pour conserver son effet dans la suite. Ce n’est pas encore une édition persistante ni un mécanisme indépendant du format de réponse.

Suite utile : explorer un déclenchement au bon moment sans position imposée, puis porter le mécanisme retenu dans un patch persistant. Cette petite réussite fournit un point de départ concret ; aucune validation indépendante n’est revendiquée.

## Bilan de session

**47 forwards sur 128**, arrêt volontaire après les comparaisons. Durée : 22 min 16,985 s ; pic mémoire rapporté : 21G, swap : 3,4G. Service terminé avec succès et processus modèle absent. Le test du hook vérifie que l’intervention reste à sa position d’origine lorsque le préfixe s’allonge ; le relevé final confirme que chaque nouveau token est l’argmax sauvegardé.

Code initial : `3f28a975`. Modèle : Gemma 4 E4B IT, vindex Q4K déjà utilisé. Session et vecteur conservés dans `/home/pat/Projects/larql-research/exploratory-natural-fact-edit/session-20260925`. Un seul journal machine (`events.jsonl`) couvre tous les essais ; `decisions.jsonl` contient aussi des rappels lors de reprises du pilote de file, sans forwards supplémentaires.

## Suite — déclenchement automatique

Nouvelle session exploratoire, même plafond de 128 forwards / 60 minutes. Le vecteur d’effet L31 reste celui de la première démonstration. Premier détecteur : similarité cosinus avec le résidu précédant `H`, seuil 0,99042 choisi entre les positifs (minimum 0,99935) et les autres étapes déjà observées (maximum 0,98149). C’est un réglage de développement, pas une validation.

Le détecteur examine la dernière position à chaque forward, sans index de génération imposé. Au premier franchissement, il applique le vecteur dans le même forward et mémorise la position absolue pour les préfixes suivants. Essais : formulation initiale, numéro atomique, hélium, puis réponse commençant par « For hydrogen: » pour déplacer le moment de la réponse. Ajustements ultérieurs autorisés dans le budget commun.
