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

## Essai suivant — transfert L21 avant les clés et valeurs partagées

26 septembre : capturer en L21 les contextes donneur hélium et receveur hydrogène de la démonstration initiale, puis reconstruire leur différence à cette couche. Tester cette direction sur le symbole de la phrase longue, gain 1 puis 2 seulement si le premier ne produit pas `He`. Conserver le premier gain qui fonctionne, ou documenter l’échec du gain 2. Rejouer la référence sans intervention avec le nouveau binaire.

Comparer ensuite les quatre cellules `H`/`He` × vecteur présent/absent sur des préfixes identiques jusqu’au nom de l’élément, et générer huit tokens libres après le symbole du candidat retenu. Position manuelle, préfixes repris et contexte donneur fourni explicitement ; aucun transfert du détecteur L31 ni revendication de déclenchement autonome en L21. 16 à 17 forwards prévus, plafond commun 24 / 25 minutes, mêmes limites mémoire. Aucun autre gain ni autre couche dans cet essai.

### Résultat L21 — propagation mesurable, édition non obtenue

**17 forwards sur 24**, service terminé avec succès le 26 septembre à 20 h 39 : 14 min 39,071 s, pic mémoire 21,6G et swap 3,6G. Les deux tests du hook passent, dont le maintien de la position avec L21/L31/L41. Les trois références recalculées avec le nouveau binaire reproduisent les anciens logits bit à bit. Les 17 tokens nouveaux sont des argmax libres ; les captures L21/L31/L41 et la différence donneur − receveur ont été vérifiées.

La direction L21 a une norme de 6,5174 pour un résidu receveur de 93,9798, soit 6,93 %. Sur le symbole de la phrase longue :

| Intervention | Symbole choisi | Logit `He` − logit `H` |
|---|---|---|
| Aucune | `H` | −17,28119 |
| L21, gain 1 | `H` | −17,24745 |
| L21, gain 2 | `H` | −17,19588 |

Les deux gains échouent donc à obtenir `He`. Le gain 2 ne déplace la marge que de +0,08531 point. Le candidat au gain 2 est conservé uniquement pour le diagnostic prévu, pas comme une édition réussie.

À préfixe fixé jusqu’au choix du nom, le même vecteur L21 modifie bien les logits d’une position ultérieure. Avec `H` dans le préfixe, la différence maximale sur le vocabulaire vaut 0,22793 et la marge hélium/hydrogène passe de −18,21708 à −18,22228 ; le modèle choisit toujours `hydrogen`. Avec `He`, la différence maximale vaut 0,15631 et cette marge passe de +1,06569 à +1,05381 ; le modèle choisit toujours `helium`. Ces effets sont faibles pour la distinction visée, mais les logits ne sont plus identiques comme dans l’ablation L31. Ce test mesure la propagation jusqu’à la sortie ; il n’instrumente pas séparément les clés et valeurs.

La continuation libre du candidat donne `In chemistry, $\text{H}$ is the symbol for hydrogen, and`, puis atteint la limite de huit tokens nouveaux. Elle ne constitue donc ni une réponse complète ni une édition obtenue. Les préfixes `He` de la matrice sont des entrées diagnostiques reprises de l’ancien essai L31, jamais des symboles attribués à une réussite L21.

**Obstacle précisé : déplacer la simple différence hélium − hydrogène vers L21 ouvre une propagation mesurable, mais ne transporte pas un effet suffisant pour changer le symbole aux gains testés.** Cela ne démontre pas que L21 est inutilisable ni qu’un gain plus élevé fonctionnerait. Aucun autre gain n’a été essayé.

Prochaine piste proposée : construire une direction à partir de deux contextes qui gardent tous deux l’hydrogène comme sujet, mais associent ce sujet à des symboles différents. Il s’agirait de tester une différence portant davantage sur la relation que sur l’identité du donneur, avec des contextes contrefactuels explicitement documentés. Cette piste reste une hypothèse, non exécutée dans cette série.

Code de départ : `9e02932e`. Traces et vecteur : `/home/pat/Projects/larql-research/exploratory-natural-fact-edit/l21-probe-20260926`. Binaire précédent conservé sous son empreinte ; nouveau binaire `230dd967e27f8070…`. Aucune modification des poids de base, aucun détecteur L21 entraîné, aucun nouvel essai après ces observations.

## Essai suivant — même sujet, association contrefactuelle

27 septembre : construire la différence L21 entre deux contextes natifs identiques sauf l’association explicite `hydrogen → H` ou `hydrogen → He`, présentée comme une convention fictive. Le préambule de réponse précédant le symbole reste celui de la phrase longue. Vérifier que la paire prédit effectivement H/He ; si elle échoue, une seconde formulation explicite de la convention est prévue. Conserver les échecs. Vérifier séparément le nom prédit dans le contexte donneur après un préfixe contenant `He`, sans attribuer cette entrée diagnostique à une génération complète.

Transférer la différence sur le prompt original sans convention, en L21 à la même position manuelle, gain 1 puis 2 si nécessaire. Comparer ensuite les logits du nom à préfixe `He` fixé, avec/sans intervention, et huit tokens libres après le symbole du candidat retenu. Le contexte fictif ne sera pas ajouté au prompt cible. Détecteur inchangé et non utilisé. Budget commun : 24 forwards / 30 minutes, mêmes limites mémoire ; aucune autre couche ni gain supplémentaire prévu.

Ajustement exploratoire pendant la série : après le recul de He au gain 1 sur la cible, ajouter un seul forward de contrôle dans le contexte source H, avec le même vecteur L21 au gain 1. Il distingue un problème de transfert entre contextes d’une dépendance au reste du contexte source. Aucun gain ni couche supplémentaire ; budget commun inchangé.

### Résultat du contraste à sujet constant

**17 forwards sur 24**, terminés normalement le 27 septembre à 12 h 59 : 16 min 22,991 s, pic mémoire 21,7G et swap 7,4G. La vérification différée après une interruption du contrôle d’approbation confirme tous les argmax, les captures, le calcul du vecteur et les deux références bit à bit. L’interruption de vérification n’avait pas arrêté le modèle.

La première formulation source suffit : deux prompts de 48 tokens, différents uniquement au token 15, prédisent respectivement `H` et `He`. Après le préfixe diagnostique contenant `He`, le donneur prédit bien `hydrogen`. Ces observations sont conditionnelles aux préfixes fournis, pas des réponses complètes depuis le prompt seul. La seconde formulation n’a pas été exécutée.

Sur le prompt cible original sans convention fictive, le nouveau vecteur L21 conserve `H` aux gains 1 et 2. La marge `He − H` recule de −17,28119 sans intervention à −17,98870 puis −18,63945. La norme du vecteur vaut 14,5610, soit 20,99 % du résidu source H. À préfixe `He` fixé, le gain 2 réduit légèrement la marge hélium/hydrogène, de +1,06569 à +1,00296, sans changer `helium`. La continuation libre du candidat reste `In chemistry, $\text{H}$ is the symbol for hydrogen, and`, tronquée à la limite prévue.

**Le contrôle supplémentaire précise l’obstacle.** Appliquer le contraste au contexte source H lui-même, gain 1, reproduit le résidu L21 du donneur He à 5,96 × 10⁻⁸ près sur la position de réponse. Pourtant, la sortie reste `H` (logits H 27,24404 ; He 9,75973), tandis que le donneur véritable choisit `He` (H 20,63326 ; He 26,31507). En L31, les résidus à cette position divergent de nouveau, avec une distance L2 de 43,64614. Le résidu L21 de cette seule position ne suffit donc pas à reproduire la décision du donneur ; le reste de l’état issu du contexte intervient. Il ne s’agit pas d’une preuve générale d’absence de représentation factuelle en L21.

Prochaine étape : capturer les autres positions du préfixe L21 dans cette paire parfaitement alignée, puis tester quelles positions doivent être transférées avant la construction des clés et valeurs L22/L23. Cette localisation restera un diagnostic sur les contextes sources, distinct d’un transfert réussi vers la question naturelle sans convention.

Code de départ : `6652dfab`. Un contrôle adaptatif additionnel est enregistré explicitement dans `decisions.jsonl` et `jobs/009-source-H-transfer.json`. Traces : `/home/pat/Projects/larql-research/exploratory-natural-fact-edit/relation-probe-20260927`. Binaire d’inférence inchangé ; environnement Python complet repris du cache après des contrôles de dépendances sans calcul modèle.

## Localisation suivante — positions du préfixe L21

Même paire source H/He alignée. Capturer toutes les lignes L21, rejouer le contrôle de la dernière position, puis tester au gain 1 : le token de convention qui diffère entre prompts ; les lignes modifiées du contexte hors dernière position ; l’ensemble des lignes modifiées si nécessaire. Lorsqu’un groupe produit He, au plus quatre essais de sous-groupes contigus cherchent un groupe suffisant plus petit. Aucune minimalité globale ne sera revendiquée. Si aucun groupe testé ne produit He, arrêt avec cet obstacle.

Sur le groupe retenu : comparer le nom à préfixe He fixé avec/sans intervention, puis huit tokens libres après le symbole. Ce diagnostic reste dans le contexte source H contenant une convention explicite, pas dans la question naturelle sans convention. Budget 24 forwards / 30 minutes, mêmes plafonds mémoire, aucun changement des poids de base. Le worker spécialisé valide les indices et dimensions des lignes ; les tests vérifient l’application aux seules positions demandées et le budget du pilote.

Ajustement exploratoire dans le budget commun : le groupe [15,16] ayant produit He tandis que 15 seul produisait H, ajouter un passage sur 16 seul pour distinguer une position suffisante d’une interaction entre les deux. Aucun changement de couche ni de gain. La continuation prévue du groupe [15,16] est conservée séparément.

Second ajustement exploratoire : le groupe [15,16] choisit He au symbole mais helium au nom. Préparer quatre contrôles du nom sur les groupes plus grands déjà positifs au symbole ([15..52], [15..33], [15..23], [15..18]). Exécution séquentielle après la continuation prévue, au plus quatre forwards supplémentaires : plafond cumulé 24 et échéance de 30 minutes depuis le début de cette localisation inchangés. Le nombre de positions suffisant au symbole ne sera pas assimilé à la préservation de l’entité.

### Résultat de localisation — deux positions suffisent au symbole, dérive du nom

Le 27 septembre, **19 forwards terminés sur 20 tentés**, plafond 24. Arrêt sur `TimeoutError: prefix diagnostic deadline` à 18 h 08, après 29 min 47,277 s ; service en échec `exit-code`, code 1. Pic mémoire 22G, swap 7,5G. La huitième étape de continuation a été interrompue. Aucun worker modèle ne reste actif. Les quatre contrôles des groupes plus grands sont préparés mais **non exécutés** : le coordinateur exigeait le succès du parent et le respect de la même échéance. Aucun budget prolongé.

Dans le contexte source H, au gain 1, les groupes [15..52], [15..33], [15..23], [15..18] puis [15,16] choisissent tous He. La dernière position seule reste H. Les positions 15 (symbole de convention) et 16 (point suivant) sont suffisantes ensemble dans ce test, sans revendication de minimalité globale.

| Positions L21 remplacées | Symbole | Marge He − H |
|---|---|---|
| Aucune | H | −17,27464 |
| 15 seule | H | −1,24960 |
| 16 seule | H | −15,62379 |
| 15 et 16 | He | +1,88706 |

À préfixe He fourni jusqu'au choix du nom, sans intervention le modèle choisit hydrogen ; avec [15,16], il choisit helium (marge helium − hydrogen : +0,11673). Les sept tokens libres sauvegardés après He donnent `In chemistry, $\text{He}$ is the symbol for helium,`. Il s'agit d'une continuation partielle dans le contexte source avec convention explicite, pas d'une édition factuelle propre sur la question naturelle sans convention.

Vérification : les 19 vecteurs de logits sont finis et leurs argmax correspondent aux tokens enregistrés ; les trois références rejouées sont identiques bit à bit. Les deltas correspondent exactement au contraste donneur − receveur et les captures des essais de symbole, à longueur identique, correspondent exactement aux lignes attendues. **Une vérification de capture échoue au seuil initial 1e−4** pour la continuation à l'étape 8 : écart maximal 0,00019073486328125 entre le préfixe capturé et celui prédit à partir de la référence plus courte. Le seuil n'est pas relevé ; ce point reste un avertissement explicite, cause non isolée. Les autres écarts de continuation sont inférieurs au seuil initial. Le résultat ne reçoit donc pas un statut de validation intégrale.

Prochaine étape préparée : comparer le nom avec les quatre groupes plus grands avant de réduire l'intervention sur le seul critère du symbole. Il faudra une nouvelle session bornée pour les exécuter. La priorité reste He avec préservation de hydrogen, puis seulement le transfert vers la question naturelle.

Code de départ : `5dbe4d17`. Traces : `/home/pat/Projects/larql-research/exploratory-natural-fact-edit/prefix-probe-20260927`. `partial-continuation.json` reconstruit seulement les tokens terminés ; aucun résultat complet n'est fabriqué pour le job interrompu. `execution-closeout.json` et `consistency-and-contrasts.json` conservent l'arrêt, les contrôles non exécutés et l'avertissement de capture.

## Reprise — groupes plus grands et conservation du nom

27 septembre, nouvelle session explicitement autorisée après clôture de la précédente. Reprendre sans modification les quatre contrôles préparés [15..52], [15..33], [15..23], [15..18], L21 gain 1. Rejouer d'abord les deux décisions de nom avec intervention absente et groupe [15,16], et vérifier les logits bit à bit. Tous les contrôles utilisent le même préfixe He fourni : ce n'est pas encore une génération autonome.

Si un groupe préserve hydrogen, sélectionner le plus petit de ces groupes positifs, revérifier son symbole puis générer huit tokens libres après He. Sinon arrêter après les six décisions. Plafond indépendant de 15 forwards et 25 minutes, MemoryHigh=22G, MemoryMax=24G, MemorySwapMax=8G ; aucun autre gain, couche ou groupe. Les anciennes traces et leur avertissement de capture restent inchangés. Un contrôle à longueur égale permettra d'évaluer les captures sans supposer leur identité avec une référence plus courte.

### Résultat — He avec hydrogène conservé dans le contexte source

**15/15 forwards, terminés normalement le 27 septembre à 23 h 20**, après 18 min 54,030 s, pic mémoire 22G, swap 4,2G. Les deux références reproduisent bit à bit les logits et toutes les captures enregistrées. Les quatre groupes plus grands préservent hydrogen au nom, à préfixe He fixé :

| Positions | Marge hydrogen − helium |
|---|---|
| 15..52 (38 positions) | +2,56452 |
| 15..33 (19 positions) | +2,07504 |
| 15..23 (9 positions) | +0,83436 |
| 15..18 (4 positions) | +0,62693 |

Le plus petit groupe testé positif, [15,16,17,18], couvre les tokens ` H`, `.`, ` Use`, ` this`. Son symbole est revérifié : **He**. Les huit tokens libres suivants donnent `In chemistry, $\text{He}$ is the symbol for hydrogen, and`. Le nom reste donc hydrogène dans cette continuation libre, contrairement au groupe [15,16]. Le préambule initial de six tokens était fourni ; le symbole et les huit suivants sont des argmax non contraints. Fragment tronqué au budget prévu, pas une réponse achevée.

Le prompt receveur reste inchangé et indique explicitement une convention fictive hydrogen → H. Les poids de base sont inchangés ; les activations L21 de quatre positions reçoivent le contraste source He − source H, gain 1. **Ce résultat montre un changement d'association avec identité préservée dans ce contexte source précis.** Il ne démontre pas encore le transfert vers une question naturelle sans convention, une édition persistante ni un déclenchement automatique. Aucun argument de minimalité globale.

Vérification des 15 argmax, des deltas et des positions appliquées. À entrées identiques, les captures L21 correspondent exactement aux lignes attendues, y compris la décision du nom en continuation. L'écart entre longueurs d'entrée différentes réapparaît à l'étape 8 : maximum 0,00019073486328125, supérieur au seuil initial 1e−4 de la série précédente. Cause toujours non isolée ; seuil non relevé et comparaisons entre longueurs traitées seulement comme descriptives. Le contrôle à longueur égale répond à une vérification différente, il n'efface pas cet avertissement.

Prochaine question : transférer l'effet vers une question sans convention explicite. Les positions [15..18] n'y ont pas de correspondance immédiate ; documenter l'alignement avant tout essai. Une piste consiste à tester d'abord les positions du préambule de réponse partagé entre contextes, au lieu de transposer aveuglément les indices du prompt. Aucun calcul supplémentaire lancé dans cette session.

Code de départ : `13a49147`. Traces : `/home/pat/Projects/larql-research/exploratory-natural-fact-edit/prefix-entity-probe-20260927`. `summary.json`, `verification.json`, `selected-intervention.json` et `report.md` conservent les résultats, leur portée et le diagnostic numérique restant.

## Transfert suivant — préambule de réponse aligné

28 septembre : tester une seule hypothèse d'alignement, sans déplacer arbitrairement les quatre positions de la convention. Les prompts source H/He (48 tokens) et la question naturelle sans convention (31 tokens) partagent les mêmes six tokens de préambule `In chemistry, $\text{`. Appliquer le contraste donneur He − receveur H déjà capturé en L21 aux six positions de ce préambule : source 48..53, cible naturelle 31..36, correspondance exacte par token et décalage depuis la fin du prompt. Gain 1 uniquement, aucune recherche de sous-groupes. Les prompts restent inchangés.

Rejouer les deux références de symbole source/naturelle et le contrôle positif des quatre positions [15..18], avec vérification des logits bit à bit. Tester ensuite le préambule dans le contexte source H et dans la question naturelle, même si le premier test échoue. Un symbole autre que He constitue un échec du transfert concerné. Si la cible naturelle reste H, arrêt après cinq forwards, sans gain supplémentaire. Si elle choisit He, comparer le nom à préfixe He fixé avec/sans intervention, puis huit tokens libres après le symbole ; helium au nom constitue une dérive d'identité, pas une édition réussie.

Nouvelle session plafonnée à 15 forwards / 25 minutes, MemoryHigh=22G, MemoryMax=24G, MemorySwapMax=8G. Vérification des captures contre leurs références à entrées identiques ; les anciens écarts entre longueurs différentes restent documentés et ne seront pas effacés par un changement de tolérance. Cette étape ne prétend pas que la différence source se transfère nécessairement entre positions absolues différentes : c'est précisément l'hypothèse testée.

### Résultat du préambule aligné — transfert négatif

**5 forwards terminés sur un plafond de 15**, arrêt normal selon la règle prévue le 28 septembre à 09 h 45. Durée 5 min 49,172 s, pic mémoire 22G, swap 3G. Les deux baselines et le contrôle positif reproduisent tous les tableaux sauvegardés bit à bit. Les cinq argmax, les deltas, l'alignement et les captures L21 à entrées identiques sont vérifiés ; erreur de capture nulle.

| Condition | Symbole | Marge He − H |
|---|---|---|
| Source H sans intervention | H | −17,27464 |
| Source H, quatre positions [15..18] | He | +1,37713 |
| Source H, préambule [48..53] | H | −17,38678 |
| Question naturelle sans intervention | H | −17,28119 |
| Question naturelle, préambule [31..36] | H | −19,91071 |

Le contraste appliqué aux six positions du préambule ne suffit pas à produire He dans le contexte source lui-même. Sur la question naturelle, il déplace la marge de −2,62952 points, dans le mauvais sens. Les deux transferts échouent au gain 1. Les contrôles de nom et la continuation conditionnés à un symbole He ne sont donc pas exécutés ; aucun ajustement de gain, couche ou groupe après le résultat.

Conclusion bornée : cette hypothèse d'alignement par préambule commun échoue pour le contraste testé. Le succès des quatre positions de la convention se reproduit, mais ne se transporte pas par cette intervention sur les seuls tokens de réponse. Cela ne prouve pas l'impossibilité d'autres transferts. La question naturelle reste sans édition factuelle obtenue ; les six tokens initiaux de réponse étaient fournis.

Prochaine piste : isoler les positions antérieures du prompt qui portent l'effet réussi avant de proposer un autre alignement. Les écarts historiques entre longueurs d'entrée restent non résolus, distincts des contrôles exacts de cette série. Aucun nouvel essai lancé après l'arrêt.

Code de départ : `ee77d3b6`. Traces : `/home/pat/Projects/larql-research/exploratory-natural-fact-edit/preamble-transfer-20260928`. `alignment.json`, `interventions.json`, `summary.json`, `verification.json` et `report.md` documentent l'hypothèse, l'échec et les contrôles.
