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

### Résultat du déclenchement automatique

**Deuxième jalon obtenu après un ajustement du détecteur.** Le cosinus initial fonctionne sur la formulation connue et ne s’active pas sur les deux contrôles. Il manque toutefois le nouveau préambule : le modèle écrit spontanément `For hydrogen: $\text{H}$` et le seuil n’est jamais franchi.

Ajustement exploratoire : apprentissage d’une direction linéaire sur les résidus normalisés, avec ridge 0,01, incluant les activations de ce nouveau format et les contrôles observés (62 lignes de développement, dont 3 positives). Seuil placé entre les scores positifs et négatifs observés. Le vecteur d’effet L31 et son gain 1 restent inchangés.

| Réponse avec le détecteur ajusté | Déclenchement |
|---|---|
| `The chemical symbol of hydrogen is **He**.` | Au 8ᵉ token, position trouvée automatiquement |
| `For hydrogen: $\text{He}$` | Au 7ᵉ token, position trouvée automatiquement |
| `The atomic number of hydrogen is **1**.` | Aucun |
| `The chemical symbol of helium is **He**.` | Aucun |

Chaque génération part ici du prompt seul, sans préambule réutilisé ni position de réponse fournie. Le détecteur regarde uniquement le résidu L31, déclenche une fois puis conserve la position trouvée. Tous les tokens restent des argmax libres et les réponses se terminent naturellement. Les scores complets disponibles des contrôles sont cette fois identiques bit à bit aux références sans intervention : 10 vecteurs pour l’hélium et 3 pour la fin de la réponse sur le numéro atomique.

**87 forwards sur 128**, arrêt volontaire après les comparaisons ; 40 min 41,021 s, pic mémoire 20,8G et swap 6,8G. Service terminé avec succès, modèle déchargé. Deux tests du hook passent, couvrant le déclenchement sur le résidu et la conservation de la position malgré l’allongement du préfixe.

Limite précise : le nouveau préambule a servi à ajuster le détecteur. Cela démontre un déclenchement autonome sur deux formats travaillés, pas une généralisation indépendante. Le prochain test utile est de garder ce détecteur fixe sur d’autres formulations et formats. L’intervention reste un prototype sur les activations ; son intégration dans une édition persistante LARQL reste à faire.

Code de départ de cette session : `f73de375`. Traces et paramètres réutilisables : `/home/pat/Projects/larql-research/exploratory-natural-fact-edit/auto-trigger-20260925`, notamment `selected-detector.json`. Aucun gain 2 n’a été exécuté et aucune nouvelle couche n’a été explorée.

## Série suivante — paramètres fixes

Sept nouveaux prompts : français, symbole seul, question indirecte via le numéro atomique, symbole dans une phrase plus longue, lithium, nombre de protons et lettre H de l’alphabet. Détecteur v2, seuil, gain et vecteur d’effet inchangés pendant toute la série. Budget global : 96 forwards, service borné à 60 minutes et mêmes limites mémoire.

Les générations éditées ont des plafonds de 8 à 20 tokens (76 au maximum au total). Lorsqu’une intervention se produit, une branche sans intervention reprend seulement le préfixe naturel précédant le déclenchement ; son budget préserve de quoi examiner les autres cas. Sans déclenchement, le forward lui-même est le chemin non modifié. Les réponses coupées par la limite seront signalées comme telles. Aucun réglage ne sera adapté à ces observations.

### Résultat des sept nouveaux cas — série close

Série terminée le **25 septembre à 21 h 23**, clôture documentaire le 26 septembre. **64 forwards sur 96**, 40 min 55,105 s, pic mémoire 21,9G et swap 3,5G. Le service s’est terminé avec succès ; aucun processus de cette expérience ne reste actif. Les sept cas prévus ont été examinés sans retouche du détecteur, du seuil, du gain ni du vecteur.

| Cas | Observation | Interprétation exploratoire |
|---|---|---|
| Symbole seul, français | Référence `H`, intervention `He`, réponses complètes | Changement obtenu ; déclenchement au premier token |
| Symbole seul, anglais | Référence `H`, intervention `He`, réponses complètes | Changement obtenu ; déclenchement au premier token |
| Élément de numéro atomique 1 | `The chemical symbol for the element with atomic number 1 is **H**.` | Échec de déclenchement ; `H` est bien atteint dans le budget |
| Phrase commençant par « In chemistry, » | `H` devient `He`, puis `hydrogen` devient `helium` | Changement de symbole accompagné d’une dérive d’entité |
| Symbole du lithium | `Li`, sans déclenchement | Contrôle non modifié |
| Protons de l’hydrogène | `1`, sans déclenchement | Contrôle non modifié |
| Huitième lettre de l’alphabet | `H`, sans déclenchement | Le détecteur ne s’active pas sur cette autre occurrence de `H` |

La phrase longue expose la limite la plus utile pour la suite :

- Référence : `In chemistry, $\text{H}$ is the symbol for hydrogen, and a chemical symbol is a`
- Intervention : `In chemistry, $\text{He}$ is the symbol for helium, and a chemical symbol is a`

Le déclenchement se produit au septième token. Le symbole change, mais la suite désigne l’hélium : ce cas ne constitue donc pas une édition isolée du symbole de l’hydrogène. La cause reste à isoler entre l’effet de l’intervention conservée dans le préfixe et les conséquences autorégressives du symbole émis. Les deux branches sont coupées à 20 tokens ; le changement d’entité est déjà observable avant cette coupure.

La formulation indirecte est coupée à 16 tokens avant l’explication demandée, mais après le symbole `H`. Son score maximal vaut −0,01716271, sous le seuil fixe −0,00592665 : c’est bien un déclenchement manqué dans la fenêtre observée. Les trois contrôles se terminent naturellement. Sans déclenchement, leurs forwards ne subissent aucune modification ; aucune branche de référence indépendante supplémentaire n’a été exécutée pour ces cas.

Vérification des traces : 64 tentatives complètes, tous les nouveaux tokens correspondent aux argmax sauvegardés, paramètres identiques à ceux du détecteur v2. Empreinte SHA-256 canonique de la spécification : `218bf2102314a3531bf0f7ac038baee035377290ab8545d303f5950a450b2a8d`. Code exécuté : `4dc1a2c352dc38267b9ab1da3e2220244ce75405`. Traces : `/home/pat/Projects/larql-research/exploratory-natural-fact-edit/fixed-probes-20260925`.

**Prochaine priorité proposée : comprendre et limiter la dérive d’entité**, puis améliorer la couverture du déclencheur. Abaisser simplement le seuil ne résout pas le problème de continuation. Aucun nouvel essai ni ajustement n’a été ajouté à cette série. Le résultat reste propre à ce prototype d’intervention temporaire sur Gemma 4 E4B Q4K ; ce n’est pas encore une édition persistante robuste.

## Diagnostic suivant — symbole émis et intervention conservée

26 septembre : comparaison croisée sur la phrase longue ayant dérivé vers l’hélium. Réutiliser les deux cellules déjà observées (`H` sans intervention ; `He` avec intervention), puis calculer les deux cellules manquantes : continuation du préfixe contenant `He` sans intervention et continuation du préfixe contenant `H` avec le même vecteur appliqué à l’ancienne position. Les huit tokens nouveaux de chaque branche couvrent la position précédemment occupée par le nom de l’élément. Deux forwards supplémentaires rejouent les cellules connues au choix du nom, afin de vérifier la comparabilité des logits sauvegardés.

18 forwards planifiés, plafond de 24 et 20 minutes, mêmes limites mémoire. Préfixes repris explicitement pour ce diagnostic ; aucune nouvelle réussite depuis le prompt seul ne sera revendiquée. Aucun ajustement du vecteur, du gain ou du détecteur. Un nom différent ou absent sera conservé comme observation, sans forcer le chemin attendu. Le diagnostic vise à distinguer les effets sur cette continuation, pas à établir un mécanisme général.

### Résultat du diagnostic — le symbole porte la dérive

**18 forwards sur 24**, terminés normalement le 26 septembre à 12 h 59 ; 14 min 45,928 s, pic mémoire 21,5G et swap 3,4G. Un premier lancement avait échoué avant tout forward, faute de module `tokenizers` dans le Python système ; le lancement effectif a repris l’environnement Python de la série précédente. Cet incident et ses journaux sont conservés.

| Symbole dans le préfixe repris | Vecteur réappliqué en L31 à l’ancienne position | Nom choisi | Logit hélium − logit hydrogène |
|---|---|---|---|
| `H` | Non | `hydrogen` | −18,21708 |
| `H` | Oui | `hydrogen` | −18,21708 |
| `He` | Non | `helium` | +1,06569 |
| `He` | Oui | `helium` | +1,06569 |

À symbole identique, les **262 144 logits sont identiques bit à bit pour chacun des huit tokens** comparés, dans les deux sens. Les préfixes restent identiques jusqu’au choix du nom dans chaque paire. Les deux relectures du token du nom reproduisent également les références bit à bit. Les deux branches nouvelles finissent à la limite prévue, après `..., and` ; ce sont des continuations diagnostiques tronquées, pas des réponses complètes depuis le prompt seul. Tous les tokens nouveaux sont des argmax libres.

Sur cette phrase, la dérive vers l’hélium vient donc du préfixe contenant `He`, sans contribution mesurable du vecteur maintenu à l’ancienne position dans la fenêtre testée. **Retirer ce vecteur après l’émission du symbole ne corrige pas la dérive.** L’écart entre les deux préfixes vaut environ 19,28277 points sur la marge hélium/hydrogène au choix du nom. Cela ne constitue pas une généralisation à d’autres prompts ou modèles.

La lecture du moteur local fournit une explication structurelle compatible avec ces observations. Ce Gemma 4 E4B possède 42 couches dont les 18 dernières partagent leurs clés et valeurs d’attention : dès L24, les couches locales les reprennent de L22, les globales de L23. Toutes les couches après L31 utilisent donc des clés et valeurs construites avant notre intervention. Le hook post-L31 modifie bien le résidu de l’ancienne position, mais cette modification tardive ne rejoint pas les clés et valeurs qu’utilisent les autres positions. C’est une déduction du code et de la configuration locale, étayée par les égalités observées, et non une nouvelle trace instrumentée des clés et valeurs.

Cette observation précise aussi le journal initial : réappliquer le vecteur avait pour intention de conserver son effet dans la suite ; **cet effet direct sur les tokens suivants n’est pas observé ici**. L’effet initial sur le symbole reste établi. Les poids de base restent inchangés.

**Prochain essai proposé : une intervention avant la construction des clés et valeurs sources, par exemple après L21**, avec un nouveau vecteur appris à cette couche et la même comparaison entre symbole émis et état interne. Une couche plus précoce ouvre une voie de propagation à tester ; elle ne garantit ni une édition plus précise ni la conservation de l’hydrogène. Aucun nouvel ajustement n’a été exécuté dans ce diagnostic.

Code de départ : `ddca5cdb`. Traces : `/home/pat/Projects/larql-research/exploratory-natural-fact-edit/continuation-ablation-20260926`. `consistency-and-contrasts.json` conserve les marges et comparaisons complètes ; `architecture-evidence.json` lie l’explication à la configuration et aux empreintes des fichiers sources. Les paramètres d’effet et le binaire correspondent exactement à la série précédente.
