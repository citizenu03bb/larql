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

## Transfert de l'effet induit — capture après L22

28 septembre, essai autorisé : reprendre exactement le prompt source H et l'intervention L21 réussie aux positions [15..18], gain 1. Capturer le préfixe complet après L22 avec et sans cette intervention. La différence provient donc de la même entrée, intervention activée moins désactivée, et non de deux prompts H/He différents. Rejouer les références existantes bit à bit avec le worker instrumenté ; celui-ci capture désormais L21 et L22 et valide une couche d'intervention 21 ou 22.

Appliquer seulement les six lignes L22 induites sur le préambule de réponse (source 48..53), gain 1, au contexte source sans patch L21. Si ce replay ne choisit pas He, arrêter après trois forwards et ne pas lancer de transfert naturel. S'il choisit He, rejouer la baseline naturelle puis appliquer ces six différences aux positions 31..36, alignées par les mêmes IDs de tokens du préambule. Si le symbole naturel n'est pas He, arrêt après cinq forwards. Sinon deux contrôles du nom à préfixe He fixé (sans/avec patch L22), puis huit tokens libres après le symbole. Un nom helium ne sera pas classé comme préservation d'identité.

Plafond total : 15 forwards / 25 minutes, MemoryHigh=22G, MemoryMax=24G, MemorySwapMax=8G. Aucun autre gain, groupe ou couche. Les poids et prompts restent inchangés. Les tests instrumentaux couvrent les positions, la capture de deux couches et l'absence d'application prématurée du patch L22. Ce test évalue la suffisance du seul état L22 de six positions ; il ne capture ni ne remplace à lui seul tout l'état d'attention ou les caches KV antérieurs.

### Résultat L22 induit — effet mesuré, replay insuffisant

**3 forwards terminés sur un plafond de 15**, arrêt normal après échec du replay source, le 28 septembre à 10 h 10. Durée 4 min 55,486 s ; pic mémoire 22G, swap 6,6G. Le transfert naturel, les contrôles du nom et la continuation ne sont pas exécutés, conformément à la règle annoncée.

Les trois tests du worker passent. Les références source sans intervention (H) et avec le patch L21 [15..18] (He) reproduisent tous les tableaux précédemment sauvegardés bit à bit. Les nouvelles captures L22 montrent un effet sur les six positions de préambule : normes par ligne 0,94609 ; 1,51161 ; 0,73930 ; 1,61832 ; 6,00244 ; 0,82496.

Appliquer ce seul effet sur L22, positions 48..53, gain 1, conserve **H**. Marge He − H : baseline −17,27464 ; contrôle L21 positif +1,37713 ; replay L22 −17,10300, soit seulement +0,17163 par rapport à la baseline. Les six états rejoués correspondent aux états du contrôle positif à 2,98 × 10⁻⁸ près. L21 reste exactement inchangée lors du replay L22 et les captures correspondent exactement à l'opération spécifiée.

Conclusion bornée : même après propagation du patch réussi vers L22, ces six états de réponse ne suffisent pas à reproduire He dans le contexte source. D'autres états du contexte ou de l'attention restent des candidats ; cette série ne les distingue pas. Le résultat naturel reste non testé pour cette intervention. Aucun réglage supplémentaire après l'arrêt.

Les trois argmax, la construction du contraste à entrée identique, l'alignement et l'application des deltas sont vérifiés. L'écart de 2,98 × 10⁻⁸ mesure la soustraction/addition face à l'état positif désiré, pas une erreur du hook. Les écarts historiques entre longueurs d'entrée différentes restent distincts et non résolus.

Code de départ : `27d6f55c`. Binaire `5c93a6bf37550c6f…`. Traces : `/home/pat/Projects/larql-research/exploratory-natural-fact-edit/l22-effect-transfer-20260928`. Le contraste complet L22 est conservé dans `induced-L22-effect.npz` ; seules ses six lignes de préambule ont été appliquées. Prochaine question : identifier quels états omis du prompt ou de l'attention doivent accompagner ces lignes avant un nouveau transfert.

## Lecture d'architecture et diagnostic de frontière L22

28 septembre : ressources primaires consultées : [Gemma 4 Technical Report, section 2](https://arxiv.org/html/2607.02770v1#S2), [configuration officielle E4B](https://huggingface.co/google/gemma-4-E4B-it/blob/main/config.json), [annonce Google](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/). Le rapport décrit les embeddings par couche, le partage KV (18 couches sur 42 pour E4B) et l'alternance locale/globale. L'optimisation K=V des autres tailles ne s'applique pas à E4B. La configuration officielle précise 42 couches, 18 partagées, hidden_size 2560 et attention_k_eq_v=false. Ces sources expliquent l'architecture, pas le mécanisme causal de notre résultat quantifié local.

Le code local place la production des K/V avant le hook post-layer. Le patch post-L22 ne change donc pas les K/V déjà produits par L22. Les couches partagées à partir de L24 reprennent L22 pour l'attention locale et L23 pour la globale (indices zéro-based). À entrée identique, les entrées PLE sont identiques aussi. Hypothèse suivante : l'état manquant peut être dans le résidu complet ou dans la mémoire KV22 réutilisée. Une ressemblance du résidu de six positions ne distingue pas ces voies.

Instrumenter la sortie KV du forward Q4K hooké : callback par défaut sans effet, après le calcul de la couche productrice et avant stockage dans le cache partagé. Le worker peut capturer puis remplacer exactement le résidu L22 complet et/ou K/V22, sans recomputation de l'attention productrice. Les remplacements restent sur les mêmes positions, avec le même prompt source H.

Six forwards seulement : baseline, contrôle positif L21 [15..18], auto-replay de la frontière baseline (résidu+KV, identité bit à bit exigée), résidu positif seul, KV positif seul, résidu+KV positifs. La combinaison est un contrôle de fermeture de frontière : vérifier si elle reproduit les logits positifs bit à bit ; un échec sera rapporté comme état encore incomplet ou défaut d'instrumentation, sans ajustement après coup. Aucun nom d'élément ni transfert naturel testé. Budget indépendant : 6 forwards / 15 minutes, mêmes plafonds mémoire 22G/24G/8G. L'approbation automatique avait échoué avant toute modification ; reprise après instruction explicite de l'utilisateur et vérification du worktree inchangé.

### Résultat de frontière — résidu et KV conjointement suffisants

**6/6 forwards, arrêt normal** le 28 septembre à 14 h 33. Durée 8 min 28,051 s, pic mémoire 22G, swap 4,5G. Cinq tests du worker et 18 tests du forward affecté passent. Les deux références reproduisent toutes leurs captures précédentes bit à bit ; l'auto-replay baseline est identique bit à bit, y compris KV et logits.

| État restauré en sortie L22 | Symbole | Marge He − H |
|---|---|---|
| Baseline | H | −17,27464 |
| Intervention originale L21 | He | +1,37713 |
| Auto-replay baseline | H | −17,27464 |
| Résidu positif complet seul | H | −2,44196 |
| KV22 positif seul | H | −14,12603 |
| Résidu positif + KV22 positif | He | +1,37713 |

**La combinaison reproduit exactement tous les logits du contrôle positif**, ainsi que les captures L31/L41. Les remplacements demandés sont exacts. Le KV22 ne diffère de la baseline qu'aux positions 15..18 ; le résidu L22 diffère aux positions 15..53. Ni l'un ni l'autre des deux remplacements isolés ne suffit à changer l'argmax, mais leur combinaison le fait. Cela établit la suffisance de cet état conjoint dans cette comparaison, pas une nécessité générale de toutes ses composantes.

La lecture du rapport et de la configuration officiels a motivé la prise en compte du cache partagé. L'explication architecturale est distincte de l'observation : le KV22 est produit avant le hook post-L22 et survit à un remplacement du résidu. Le résidu positif change aussi les calculs ultérieurs, dont la production des KV globaux en L23 ; cette série ne sépare donc pas chaque voie d'attention, ni K de V.

Portée inchangée : contexte source avec convention explicite, décision du symbole après préambule fourni. Aucun nom d'élément ni question naturelle sans convention testé. Les anciens écarts entre longueurs restent non résolus ; les présents contrôles comparent exactement les mêmes entrées. Aucun nouveau calcul après les six passes. Prochaine question : garder fixes les quatre positions KV changées et distinguer les positions de prompt et de réponse dans le résidu qui doit les accompagner.

Code de départ : `fc100b89`. Binaire `d5138ff5f3e889ee…`. Traces : `/home/pat/Projects/larql-research/exploratory-natural-fact-edit/kv-boundary-20260928`. `architecture-sources.json`, `summary.json`, `verification.json` et `report.md` relient les sources, le protocole et les observations.

## Localisation résiduelle avec KV22 positif fixé

28 septembre, suite autorisée : conserver exactement le KV22 positif de la série précédente (seules les positions 15..18 diffèrent de la baseline) et varier uniquement les lignes du résidu L22. Le binaire hashé reste `d5138ff5f3e889ee…`, sans modification du worker ni du forward.

Six passes fixées avant observation : références KV seul (H) et frontière combinée complète (He), avec replays bit à bit exigés, puis remplacement positif du résidu sur (1) les positions modifiées du prompt 15..47, (2) le préambule de réponse 48..53, (3) les quatre positions originales 15..18, (4) le reste du prompt modifié 19..47. Toutes les autres lignes résiduelles restent à la baseline et le KV reste positif. Les groupes prompt/réponse forment une partition des 39 lignes modifiées ; les quatre positions et le reste forment une partition du prompt modifié. Aucun ajustement de groupe après résultat.

Budget : 6 forwards / 15 minutes, MemoryHigh=22G, MemoryMax=24G, MemorySwapMax=8G. Les remplacements portent sur des captures de même longueur et sont exacts. La décision du symbole est le seul critère : He indique la suffisance du groupe testé avec ce KV fixé, sans minimalité globale ni conclusion sur le nom de l'élément. Aucun transfert naturel ou continuation dans cette session.

## Résultat des groupes résiduels — clôture du 21 octobre 2026

Essai du 28 septembre 2026, clôturé et vérifié le 21 octobre 2026. Les six forwards prévus sont terminés ; le runner a enregistré un code de sortie 0. Aucun nouveau calcul modèle lancé lors de cette clôture.

| Résidu positif restauré, avec KV22 positif constant | Symbole | Marge He − H |
|---|---|---:|
| Aucun (référence KV seul) | H | −14,12603 |
| Toutes les positions modifiées 15..53 | He | +1,37713 |
| Prompt 15..47 | He | +1,33128 |
| Préambule de réponse 48..53 | H | −13,87466 |
| Quatre positions originales 15..18 | H | −0,71773 |
| Reste du prompt 19..47 | H | −9,84135 |

Le résidu des positions du prompt, accompagné du KV22 positif, suffit au choix de He dans ce contexte source. Restaurer les six états du préambule de réponse n'est pas nécessaire à ce choix. Les deux sous-groupes du prompt testés séparément ne suffisent pas ; leur union réussit. Cela ne démontre ni minimalité, ni nécessité générale, ni mécanisme sémantique universel.

Vérifications : les deux références reproduisent tous leurs tableaux bit à bit ; les six argmax sont confirmés ; les remplacements résiduels correspondent exactement aux groupes annoncés ; KV22 reste exactement positif et L21 exactement baseline dans les six cas ; formes et valeurs finies vérifiées ; SHA-256 du binaire conforme. Vérification exécutée avec Python système et NumPy 2.3.5, l'ancien environnement uv n'étant plus présent.

Budget annoncé : six forwards, 15 minutes, MemoryHigh=22G, MemoryMax=24G, MemorySwapMax=8G. Le service transitoire n'est plus enregistré et son journal n'est plus disponible à la clôture : durée effective et pics mémoire ne sont donc pas attestés ici. La terminaison est établie par les événements complets, summary.json et l'absence actuelle de worker, pas par le Result=success par défaut d'une unité absente.

Portée : Gemma 4 E4B Q4K, prompt source contenant une convention fictive explicite, six tokens de préambule fournis, décision d'un seul symbole. Ni préservation du nom dans la continuation, ni transfert à une question naturelle, ni patch persistant établis par cette série. Les écarts historiques entre longueurs d'entrée restent à distinguer de ces contrôles à entrée identique.

Recommandation : continuer avec une seule série courte visant le transfert à une question naturelle sans convention, en explicitant l'alignement des positions et le traitement du KV dépendant de la position. Prévoir baseline, contrôle positif source et contrôle d'identité du transfert avant d'interpréter un résultat négatif. Si le symbole change, examiner la continuation et quelques questions voisines ; sinon identifier la limite technique observée avant tout nouvel ajustement. Éviter une nouvelle longue recherche de sous-groupes qui ne rapprocherait pas de cet objectif. Cette prochaine série n'est pas lancée.

Code du protocole : e6ef45ee7d528bfe073ef5ff4fbc3be112c67af4. Binaire : d5138ff5f3e889eeb7c00356370468fc6747d192ba044d3ff1bfc14c7646b61c.

Traces : /home/pat/Projects/larql-research/exploratory-natural-fact-edit/kv-residual-groups-20260928 ; vérifications détaillées dans verification.json.

## Transfert vers deux consignes neutres à positions fixes — 21 octobre 2026

Suite autorisée : remplacer les 17 tokens de convention fictive du prompt source par deux consignes neutres de 17 tokens chacune, sans modifier les positions 21..47 de la question et du format chat. Les prompts exacts sont dans `scripts/run_neutral_boundary_transfer.py` ; leur encodage aller-retour est vérifié. Même longueur 48 tokens et préambule de réponse fourni de six tokens. Le KV capturé contient déjà RoPE : aucune translation ni rotation n'est nécessaire ici. Les positions du préfixe sont alignées par coordonnée, pas par identité lexicale. Ce test ne vaut donc pas transfert à toute paraphrase ni à la question courte de 31 tokens.

Intervention : différence positive moins baseline, issue des captures source à entrée identique, gain 1. Ajouter le delta L22 aux lignes de prompt 15..47 de la cible, et les deltas K/V22 aux lignes 15..18. Toutes les autres lignes, notamment le préambule de réponse, restent à l'état natif cible. Il s'agit d'un delta ajouté aux états cibles, non d'un remplacement par le contexte source complet. Aucun poids modifié.

Huit passes initiales : replay positif source bit à bit ; baseline neutre A ; replay identité A bit à bit ; transfert conjoint A ; résidu seul A ; KV seul A ; baseline neutre B ; transfert conjoint B. Chaque remplacement et L21 inchangé sont vérifiés. Si aucun transfert conjoint ne choisit He, arrêt à huit passes sans réglage supplémentaire. Sinon continuation du premier succès (A prioritaire) pendant au plus huit tokens : pour chacun, capture native à l'entrée exacte actuelle puis nouvelle passe avec le delta fixe, soit deux passes par token. Cette construction évite de supposer l'égalité des captures entre longueurs différentes. La continuation ne constitue pas un test de questions voisines.

Plafond total : 24 forwards / 30 minutes, MemoryHigh=22G, MemoryMax=24G, MemorySwapMax=8G, un seul worker. Binaire inchangé d5138ff5f3e889ee… ; Python local recréé avec NumPy 2.3.5 et tokenizers 0.23.3. Une erreur de contrôle arrête la série. Les résultats seront exploratoires et distingueront symbole, nom dans la continuation et généralisation.

### Résultat du transfert neutre — huit passes, aucun basculement

21 octobre : huit passes terminées, code de sortie 0, arrêt conditionnel normal. Durée systemd 9 min 1,176 s ; pic mémoire 22G, swap 5,2G. Le contrôle positif source reste He et se reproduit bit à bit. Le replay identité naturel A est aussi bit-identique. Tous les deltas, matrices et argmax sont vérifiés indépendamment.

| Condition | Symbole | Marge He − H |
|---|---|---:|
| Contrôle positif source | He | +1.37713 |
| Neutre A, baseline | H | -15.99848 |
| Neutre A, identité | H | -15.99848 |
| Neutre A, résidu + KV | H | -15.65666 |
| Neutre A, résidu seul | H | -15.86723 |
| Neutre A, KV seul | H | -15.78289 |
| Neutre B, baseline | H | -16.07501 |
| Neutre B, résidu + KV | H | -16.06973 |

Le transfert conjoint produit seulement +0,34181 sur A et +0,00528 sur B : H reste le choix dans les deux cas. Aucune continuation ni question voisine supplémentaire exécutée. Le contraste source au gain 1 est insuffisant sur ces deux formulations neutres à positions fixes ; aucun défaut d’application détecté. La cause exacte reste ouverte, notamment l’alignement seulement positionnel des mots du préfixe et les états omis. Aucun poids du modèle modifié, aucun patch persistant créé.

Code de lancement 0b87c9d7 ; traces neutral-boundary-transfer-20261021 sous le répertoire de recherche. Prochaine piste proposée, non lancée : construire une intervention dans le contexte cible lui-même avec contrôle de préservation de l’entité. Rapport détaillé et vérification archivés.

## Contraste natif de la question cible — 21 octobre 2026

Suite autorisée : question naturelle courte de 31 tokens, six tokens de début de réponse fournis. Donneur construit par l'unique remplacement hydrogen→helium au token 14 ; les deux questions ont exactement la même longueur et le même format. Capturer L21, calculer donneur moins baseline et intervenir sur la cible hydrogen, sans modifier ses tokens ni les poids. Le forward recalcule ensuite ses caches KV normalement.

Quinze passes initiales : baseline cible avec replay historique bit à bit, donneur attendu He, contrôle gain zéro bit à bit, puis trois groupes ([17] relation, [14] entité, [14..24] segment entité/relation) à chacun des gains 1, 2, 4 et 8, dans cet ordre croissant de gain. Chaque application L21 est vérifiée exactement à entrée égale. Aucun réglage supplémentaire après ces douze candidats. Premier candidat He sélectionné selon cet ordre, puis au plus huit tokens libres de continuation avec la même intervention ; plafond total 23 forwards / 30 minutes, limites mémoire 22G/24G/8G, worker unique inchangé.

Le donneur helium introduit délibérément un risque de substitution d'entité : He seul n'est pas un succès factuel. La continuation doit être examinée pour distinguer une phrase sur hydrogen d'une phrase sur helium. Les huit tokens peuvent rester insuffisants pour conclure si le nom n'apparaît pas. Aucune généralisation, patch persistant ou génération sans préambule fourni n'est établie par cette série. Le contraste natif remplace les tentatives de transfert depuis la convention fictive.

### Résultat du contraste natif — substitution d’entité pour le candidat suivi

23/23 passes vérifiées, arrêt normal le 21 octobre. Durée 17 min 31,765 s ; pic mémoire 22G, swap 3,1G. Baseline historique et contrôle gain zéro bit-identiques ; douze applications L21 exactes, tous les argmax et la chaîne de continuation vérifiés.

Aux gains 1/2/4/8, le token relation [17] reste H ; l’entité [14] et le segment [14..24] donnent He, soit 8/12 candidats. Marges He−H du token relation : −17,69336 / −18,01808 / −18,15211 / −17,29992 ; entité : +11,49983 / +19,46988 / +19,17976 / +12,52797 ; segment : +12,31189 / +20,60350 / +20,31885 / +13,42689. Groupes non normalisés en amplitude.

Premier candidat He retenu : entité seule au gain 1. Continuation libre de huit tokens après le symbole : `In chemistry, $\text{He}$ is the symbol for helium, and`. L’identité n’est pas préservée : pas d’édition factuelle réussie pour ce candidat. Les sept autres continuations restent non testées. Six tokens initiaux de réponse étaient fournis ; aucune reformulation ni question témoin supplémentaire.

Les requêtes de continuation sont vérifiées, sans capture baseline appariée à chaque longueur plus grande. Code eab503f4, traces target-context-probe-20261021. Aucun poids modifié. Prochaine piste, non lancée : groupes après l’entité sans toucher le token hydrogen, avec contrôle du nom. L’échec du seul token 17 ne les exclut pas.

## Poursuite de bout en bout — recherche après l’entité

21 octobre, autorisation explicite de poursuivre jusqu’à validation. Première série : contraste natif L21 hydrogen/helium déjà défini ; préserver le token 14 et tester les groupes 15..24, 18..30 et 31..36 aux gains 1/2/4. Baseline, donneur et gain zéro sont rejoués ; un contrôle du nom à préfixe He fixé sert de référence. Pour chaque symbole He, un probe du nom vérifie si hydrogen reste top1 ; premier candidat passant les deux critères retenu pour une continuation libre dans la série suivante. Le probe à préfixe fourni ne sera pas présenté comme une génération libre. Maximum 22 passes utiles, plafond 24 / 30 minutes, mémoire 22G/24G/8G. Pas de modification des poids. Si aucun candidat ne passe, examiner séparément le maintien de l’identité après l’émission du symbole, sans confondre contrôle temporel des activations et édition persistante.

### Résultat après l’entité et prochaine intervention temporelle

13/13 passes, contrôles exacts. Les neuf candidats restent H ; marges He−H de −18,30220 à −15,69027. Aucun probe du nom conditionnel ni continuation. Token 14 inchangé. Durée 10 min 30,577 s, mémoire 22G / swap 7,6G. Traces post-entity-probe-20261021.

Suite : contrôle dynamique distinct. Rejouer baseline naturelle et candidat symbole positif, puis tester sur le nom à préfixe He fixé les gains d’entité −0,125 / −0,25 / −0,5 / −1, en retenant le premier hydrogen. Si succès, générer les six premiers tokens depuis une réponse vide ; exiger le préambule calibré avant intervention. Produire He avec gain +1 puis générer huit tokens avec le gain négatif choisi. Aucun token de la démonstration finale ne sera fourni. Si le préambule natif diffère, arrêter sans injecter au mauvais endroit. Plafond 24 passes / 30 minutes, mêmes limites mémoire, poids inchangés.

### Résultat temporel — échec du gain inverse sur le nom

Exécuté le 21 octobre, vérifié le 22 octobre. Sept passes, arrêt normal : trois contrôles et quatre gains négatifs (−0,125 / −0,25 / −0,5 / −1). Tous produisent helium ; marges hydrogen−helium −1,20919 / −1,46430 / −2,20946 / −4,18083, contre −1,06569 sans intervention. Le gain inverse dégrade donc ce critère dans la configuration testée. Aucun gain retenu, aucune génération depuis une réponse vide ni continuation temporelle finale exécutée.

Replays bit-identiques, applications exactes, sept argmax vérifiés. Durée 6 min 42,287 s ; mémoire 22G, swap 2,7G. Code 61640dec ; traces temporal-identity-probe-20261021. Poids inchangés. L’objectif d’édition préservant l’entité reste non atteint ; ne pas confondre la terminaison normale de la calibration avec une validation du contrôleur à deux phases.

## Entité avant symbole — 22 octobre 2026

Suite approuvée : question « What is hydrogen's chemical symbol? Answer in one short sentence, naming the element before its symbol. » Réponse vide, génération greedy native. À la première proposition du token H après que hydrogen apparaît dans le texte généré (12 propositions maximum), suspendre l'émission de ce symbole. Capturer le donneur helium au même préfixe ; contraste L21 au token d'entité du prompt (position 6), contrôle gain zéro exact, gains 1/2/4 jusqu'au premier He greedy. Le token H du calcul baseline est une proposition diagnostique, pas un token émis puis remplacé dans la branche éditée. Aucun logit ni token n'est forcé.

Si He est obtenu, générer une courte branche témoin avec le H natif (4 tokens maximum, en réservant 4 passes), puis la branche He sans plus aucune intervention jusqu'à EOS ou au plafond. Rapporter séparément toute troncature de branche ; une phrase partielle ne sera pas déclarée comme absence de correction. Plafond total 24 forwards / 30 minutes, mêmes limites mémoire 22G/24G/8G. Il s'agit d'un contrôleur local conditionné à une proposition H et au nom déjà généré, non d'une mise à jour permanente des poids ni d'un détecteur général validé.

### Résultat entité avant symbole — premier jalon observable

14/14 passes vérifiées. Depuis une réponse vide, le modèle génère « Hydrogen's chemical symbol is ». Gain 1 sur le token d’entité L21 au moment du symbole : H→He ; intervention ensuite relâchée. Branche témoin complète : « Hydrogen's chemical symbol is H. » ; branche éditée complète : « Hydrogen's chemical symbol is He. ». Les deux atteignent la fin de tour. Aucun token fourni ni logit forcé. Contraste/application exacts, identité gain zéro bit-identique, chaînes greedy vérifiées. Durée 9 min 47,443 s, mémoire 22G, swap 4,8G. Code 2eb63239, traces entity-first-probe-20261022.

Il s’agit d’un contrôleur local à proposition H après le nom généré et recalcul ponctuel, pas d’une édition persistante. Suite autorisée : conserver le vecteur et le gain, tester deux reformulations depuis une réponse vide (10 passes maximum chacune), puis deux paires baseline/contrôleur non éligible pour helium et oxygen à préfixe fourni. Les contrôles vérifient la règle de non-déclenchement, pas un détecteur sémantique appris. Plafond distinct 24 passes / 30 minutes, mêmes limites mémoire ; toute troncature sera rapportée.

### Reformulations à vecteur fixe — clôture du 23 octobre 2026

Les 24/24 forwards se sont terminés normalement. Vérification indépendante réussie : vecteur et gain gelés, départ de réponse vide, argmax, chaînes, applications L21 exactes et contrôles bit-identiques. Durée systemd 16 min 7,116 s, mémoire 22G, swap 6,3G.

Les deux reformulations ont remplacé la proposition H par un début de formatage LaTeX. Elles atteignent leur plafond respectif avant le symbole :

- `Hydrogen's chemical symbol is $\text{`
- `Hydrogen is represented by the symbol $\text{`

L'effet immédiat diffère donc du He obtenu sur la question source. Le résultat factuel des réponses complètes est encore indéterminé ; ces troncatures ne prouvent ni un transfert réussi ni son échec sémantique. Les contrôles helium et oxygen à préfixe fourni restent bit-identiques sans déclenchement, respectivement He et O. Ils vérifient la règle lexicale du contrôleur, pas une sélectivité sémantique apprise.

Aucun poids, vindex ou patch persistant modifié. Traces : entity-first-neighbors-20261022. Code du lancement : 8d06e1f6.

## Compléter les reformulations tronquées — 23 octobre 2026

Nouvelle série bornée, distincte du plafond précédent. Pour chacune des deux reformulations, reprendre exactement les IDs générés sauvegardés du dernier job (009 / 019), puis poursuivre sans aucune intervention jusqu'à EOS ou huit tokens supplémentaires. Reprendre également la proposition H native de la frontière sauvegardée (006 / 016) pour obtenir une branche témoin, jusqu'à EOS ou quatre tokens supplémentaires. Tous les préfixes proviennent des branches précédemment générées depuis une réponse vide ; aucun texte réencodé ni nouveau symbole fourni. Leur provenance sera vérifiée par hash et chaîne d'IDs.

Critère : examiner les deux phrases complètes et le symbole réellement choisi après le formatage. Si H, le premier déclenchement a perturbé le formatage sans transférer le fait ; si He, inspecter la fin de tour et toute correction éventuelle. Toute troncature reste explicitement indéterminée. Aucun ajustement de gain, nouvelle injection ou recherche de donneur dans cette série. Maximum 24 forwards / 30 minutes, un seul worker, mémoire 22G/24G/8G, binaire inchangé.

## Seconde application au symbole formaté — protocole exploratoire du 23 octobre 2026

Les deux premiers tokens libres après les préambules LaTeX sauvegardés sont H. Une série distincte testera le même vecteur L21 et le même gain 1 à cette nouvelle frontière, sous réserve de clôture vérifiée des quatre branches de continuation. Le vecteur reste celui de la question source, projeté au token hydrogen de chaque question ; aucun nouveau donneur ni gain ajusté. La chaîne générée jusqu'à cette frontière contient déjà l'intervention initiale qui a produit le formatage. Ce test porte donc sur un contrôleur révisé à deux déclenchements, pas sur le transfert à déclenchement unique initial.

Pour chaque reformulation : replay natif apparié à la capture historique du H formaté ; contrôle gain zéro bit-identique ; application exacte du vecteur gelé ; émission de l'argmax quel qu'il soit ; relâchement et neuf tokens maximum jusqu'à EOS. Maximum 12 forwards par reformulation, 24 au total / 30 minutes, mémoire 22G/24G/8G, worker unique. Aucun texte de réponse retapé ou symbole forcé. Succès exploratoire seulement si la réponse complète préserve hydrogen et affirme He sans correction. Une nouvelle troncature reste indéterminée. Ces deux reformulations ont déjà été observées et ne constituent pas une validation indépendante.

### Résultat de la complétion sans intervention

10 forwards terminés et vérifiés. Les préfixes proviennent exactement des réponses générées dans entity-first-neighbors-20261022, avec vérification des IDs et hashes. Aucun préfixe retapé, aucune intervention additionnelle, tous les argmax vérifiés.

| Reformulation | Branche | Réponse complète | Fin de tour |
|---|---|---|---|
| 1 | edited | `Hydrogen's chemical symbol is $\text{H}$. [fin de tour]` | True |
| 1 | baseline | `Hydrogen's chemical symbol is H. [fin de tour]` | True |
| 2 | edited | `Hydrogen is represented by the symbol $\text{H}$. [fin de tour]` | True |
| 2 | baseline | `Hydrogen is represented by the symbol H. [fin de tour]` | True |

Les deux interventions initiales ont changé le formatage mais pas le fait : le symbole écrit est H dans les deux réponses éditées complètes, comme dans les témoins. Le transfert à déclenchement unique est négatif sur ces deux reformulations. La démonstration source Hydrogen→He du 22 octobre reste valide ; elle ne se généralise pas telle quelle.

La prochaine expérience isole le moment d'application : seconde application du même vecteur au H réellement écrit dans la formule. C'est un nouveau contrôleur exploratoire, pas une réinterprétation positive du test précédent. Aucun poids, vindex ou patch persistant modifié.

Durée systemd de cette complétion : 8 min 52,069 s ; pic mémoire 22G, swap 5,3G. Dix passes, quatre fins de tour, code de sortie 0.

### Résultat de la seconde application

Dix passes terminées et vérifiées. Replays baseline et contrôles gain zéro bit-identiques ; vecteur source, gain 1, positions, applications L21, argmax et continuations sans intervention vérifiés. Durée systemd 8 min 53,159 s ; mémoire 22G, swap 6,4G.

- Reformulation 1 : `Hydrogen's chemical symbol is $\text{H}$.<turn|>` — échec.
- Reformulation 2 : `Hydrogen is represented by the symbol $\text{He}$.<turn|>` — modification factuelle observable, hydrogen préservé, fin de tour atteinte sans correction.

Le vecteur est exactement celui de la démonstration source, sans recalibration. La seconde reformulation réussit avec deux applications ponctuelles : une première qui produit le formatage, une seconde au H à l'intérieur de la formule. Le contrôleur original à déclenchement unique reste négatif sur les deux reformulations ; le contrôleur révisé réussit sur une des deux. Elles ont été observées pendant l'exploration et ne sont pas un holdout indépendant.

Les chaînes remontent au départ de réponse vide de la série précédente : les préfixes sauvegardés proviennent uniquement des tokens émis. La chaîne est recomposée et vérifiée entre plusieurs exécutions, pas rejouée ici dans une unique exécution depuis zéro. Aucun token He ni logit forcé. La baseline complète de la seconde formulation donne H. Aucun poids, vindex ou patch persistant modifié.

Le moment du déclenchement explique une partie des limites, mais pas l'échec de la première formulation. Un contraste propre à cette première question sera testé au même gain 1 à sa frontière initiale, pour distinguer adaptation au contexte et transfert du vecteur gelé.

## Contraste local sur la reformulation encore négative — 23 octobre 2026

Un seul cas sélectionné : reformulation 1 (index 0), toujours H après les deux applications du vecteur gelé. Reprendre sa frontière initiale de proposition H, dont le préfixe est issu de la génération vide ; replay baseline exact, donneur modifiant uniquement le token hydrogen du prompt en helium, différence donneur−baseline L21 au token d'entité position 5, contrôle gain zéro exact, application au gain 1, puis relâchement jusqu'à EOS ou huit tokens. Maximum 12 passes / 30 minutes, mémoire 22G/24G/8G, un worker. Aucun gain ni position recherchés.

Comparer au résultat gelé à cette même frontière, qui produisait un préambule LaTeX puis H. Cette adaptation vise un diagnostic de dépendance au contexte, pas une nouvelle mesure de généralisation. Rapporter le symbole, la phrase entière et toute troncature ; comparer aussi direction et norme des vecteurs sans en déduire à elles seules une causalité. Cette série clôturera la boucle actuelle avec une démonstration ou un obstacle explicite ; aucune recherche ouverte supplémentaire prévue dans ce budget.

### Résultat du contraste adapté

9 calculs terminés et vérifiés. Réponse : `Hydrogen's chemical symbol is $\text{H}$.<turn|>`. Fin de tour : True. Symbole proposé après intervention : ` $\`.

La baseline à cette même frontière reproduit exactement le H historique. Le donneur ne diffère que par hydrogen→helium au token de prompt 5 ; son symbole proposé est ` He`. Le delta L21 donneur−baseline est calculé puis appliqué uniquement à cette ligne au gain 1. Contrôle zéro bit-identique, application exacte, tous les argmax et la continuation relâchée vérifiés. Les tokens précédents proviennent de la génération depuis une réponse vide du 22 octobre, avec provenance contrôlée ; ils n'ont pas été retapés.

Norme du nouveau vecteur : 29.669364 ; cosinus avec le vecteur source gelé : 0.951097. Ces mesures décrivent leur différence et ne suffisent pas seules à expliquer la causalité. Ce cas a été sélectionné après observation d'un échec ; il s'agit d'adaptation locale, pas de validation indépendante ni de transfert du vecteur gelé. Aucun poids, vindex ou patch persistant modifié.

Durée du contraste local : 8 min 13,447 s ; mémoire 22G, swap 5,8G. Neuf passes, terminaison normale. Résultat complet H malgré le nouveau contraste (cosinus 0,951097 avec le vecteur gelé).

## Extension autorisée : états distribués — 23 octobre 2026

L'utilisateur réitère « continue de bout en bout » et demande de signaler honnêtement une impasse. Le bilan précédent à 29 passes est un point d'étape ; la boucle est prolongée par ce diagnostic borné. Les succès observés excluent de parler d'impossibilité générale, mais le vecteur isolé à L21 reste insuffisant sur la première reformulation, même recalculé localement.

Même question, même préfixe généré, même contraste local L21 et gain 1. Trois contrôles (replay baseline, replay donneur, zéro), puis quatre groupes fixés : prompt depuis l'entité [5..29] ; prompt après l'entité [6..29] ; réponse déjà générée [30..35] ; toutes les lignes depuis l'entité [5..35]. Les bornes sont calculées à partir des IDs sauvegardés. Chaque application est vérifiée exactement. Choisir le premier He immédiat dans cet ordre pour huit tokens de continuation relâchée maximum ; si aucun He, suivre le premier groupe comme diagnostic de formatage. Les autres continuations ne seront pas prétendues testées. Maximum 15 passes / 30 minutes, un seul worker, mémoire 22G/24G/8G.

Attention d'architecture vérifiée localement : index.json déclare per_layer_embed_dim=256 ; remote_ffn.rs pré-calcule les entrées PLE depuis les embeddings et token_ids avant la boucle de couches et les transmet à chaque couche. Le hook L21 ne les remplace pas. Même la condition toutes-lignes-L21 ne constitue donc pas une transplantation de l'état modèle complet ni un replay donneur garanti. Le PLE est une voie non manipulée, pas une cause prouvée des échecs.

### Résultat des groupes L21

Neuf passes terminées et vérifiées. Baseline/donneur et gain zéro bit-identiques ; quatre applications exactes, ordre de sélection, argmax et continuation vérifiés. Durée 8 min 9,032 s ; mémoire 21,7G, swap 7G.

| Groupe, gain 1 | Proposition à la frontière | Continuation |
|---|---|---|
| Prompt de l'entité à la fin, 5..29 | He | `Hydrogen's chemical symbol is He.` puis fin de tour |
| Prompt après l'entité, 6..29 | Début LaTeX | Non testée |
| Réponse déjà générée, 30..35 | H | Non testée |
| Toutes les lignes depuis l'entité, 5..35 | He | Non testée |

Le premier He dans l'ordre annoncé est retenu : groupe prompt, gain 1. Le contraste est recalculé dans cette question ; il n'est pas le vecteur gelé source. Une application ponctuelle à la frontière du symbole, puis aucun patch dans la continuation. Les préfixes étaient réellement générés lors de la série initiale mais repris depuis les traces : une exécution intégrée depuis zéro est la prochaine vérification.

Ce résultat résout l'obstacle local de la position unique sur cette formulation. Il ne prouve pas qu'un support large soit nécessaire : les groupes n'ont pas la même énergie. Norme prompt 76,69817 contre entité seule 29,66936 (ratio 2,58510). Localisation et amplitude sont confondues ; les continuations des trois autres groupes restent non testées.

La condition toutes-lignes retrouve le L21 donneur à 2,38419e-7 près mais ses logits diffèrent jusqu'à 1,94086. Les PLE natifs restent non remplacés et les effets numériques ne sont pas isolés : ne pas attribuer cet écart à une cause unique. Aucun poids, vindex ou patch persistant modifié ; aucune impasse générale établie.

## Vérification intégrée depuis une réponse vide

Même première reformulation observée, départ de réponse vide dans un nouveau service. Génération native jusqu'à la proposition H après hydrogen généré, au plus 12 propositions. Recalculer le contraste donneur helium dans ce contexte exact ; contrôle zéro ; une seule application du groupe prompt depuis le token d'entité jusqu'à la fin du prompt, gain 1. Groupe choisi et gain fixés par le résultat précédent, pas de recherche additionnelle. Si l'argmax n'est pas He, rapporter l'échec du replay sans réglage. Sinon continuer la baseline H (au plus quatre tokens) et la branche He sans intervention jusqu'à EOS ou budget. Aucun préfixe de réponse fourni ni logit/token forcé. Maximum 24 forwards / 30 minutes, mêmes limites 22G/24G/8G, worker unique. Comparer les captures aux essais précédents et vérifier toutes les chaînes depuis zéro.

### Résultat de la vérification intégrée

Succès local vérifié : `Hydrogen's chemical symbol is H. [fin de tour]` devient `Hydrogen's chemical symbol is He. [fin de tour]`.

14 forwards dans une seule exécution depuis une réponse vide. La question est : « Give hydrogen's chemical symbol in one short sentence, with the element's name before the symbol. » Aucun préfixe de réponse fourni, aucun token ni logit forcé. Les tokens du préfixe sont générés naturellement ; à la proposition H, le contraste hydrogen/helium est calculé au même préfixe. Une application à L21, positions de prompt 5..29, gain 1, puis relâchement complet. Les branches témoin et éditée atteignent toutes deux la fin de tour.

La vérification indépendante reproduit bit à bit la baseline, le donneur et le patch de la série diagnostique, vérifie le gain zéro, chaque application, les argmax, le départ vide et les deux chaînes complètes. Le groupe et le gain étaient fixés avant cette exécution. Durée 10min 10.876s, mémoire 22G, swap 5.8G.

Cette formulation auparavant négative fonctionne maintenant dans une exécution intégrée. C'est une intervention d'activation locale avec contraste adapté à la question ; aucune mise à jour des poids, du vindex ou d'un patch persistant. Ce replay utilise une question déjà observée et ne constitue pas une validation sur un jeu intact.

## Étendue contre amplitude — 23 octobre 2026

Suite approuvée après recommandation de commencer par une seule série de 24 passes / 30 minutes. Même première reformulation, même frontière H issue du préfixe généré et même contraste local L21. Comparaison 2×2 : position d'entité 5 versus segment prompt 5..29 ; norme Frobenius basse N_entité=29,66936 et haute N_prompt=76,69817. Cases connues : entité gain 1 et prompt gain 1, rejouées bit à bit. Cases nouvelles : entité gain N_prompt/N_entité≈2,58510 et prompt gain N_entité/N_prompt≈0,38683. Gains arrondis float32 comme le moteur ; normes des deltas réellement transmis égales à la cible à 1e-6 relatif près.

Six passes de frontière : baseline historique, zéro sur le segment prompt, deux replays connus, puis deux nouveaux candidats. Chacune des deux nouvelles branches émet son argmax quel qu'il soit et poursuit sans intervention, jusqu'à EOS ou neuf tokens supplémentaires. Maximum 24 forwards, worker unique, MemoryHigh=22G/MemoryMax=24G/MemorySwapMax=8G, 30 minutes. Les continuations historiques des deux cases connues sont conservées après vérification du replay ; aucun préfixe n'est retapé. Cette série est un diagnostic à frontière reprise, pas une nouvelle génération intégrale depuis zéro.

L'égalité de norme des perturbations ne garantit pas une égalité d'effet dans le modèle non linéaire. Si l'entité amplifiée réussit, le support large n'est pas nécessaire sur ce cas. Si le segment atténué réussit mais pas l'entité amplifiée, la seule norme globale n'explique pas la différence. Si seules les lignes prompt à forte norme réussissent parmi ces quatre cases, les deux facteurs restent liés à la réussite dans le domaine testé. Toute troncature ou correction sera rapportée ; aucune nouvelle formulation ni généralisation n'est mesurée ici. Aucun poids ou patch persistant modifié.
