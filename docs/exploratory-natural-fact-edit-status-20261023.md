# LARQL — bilan intermédiaire du 23 octobre 2026

La démonstration naturelle du 22 octobre est conservée et une seconde formulation fonctionne avec le même vecteur, à condition de réappliquer l'intervention au symbole après le formatage. Le contraste adapté ne satisfait pas le critère complet He avec hydrogen dans ce cas. La limite demeure explicite sur cette première reformulation.

## Résultats observés

| Question / condition | Une application du vecteur source | Seconde application au symbole formaté | Contraste adapté à la question |
|---|---|---|---|
| Question source, entité avant symbole | **He**, phrase complète | Non nécessaire | Vecteur construit sur cette question |
| Reformulation 1, « Give hydrogen's chemical symbol… » | H en LaTeX, phrase complète | H en LaTeX, phrase complète | H, fin de tour : True |
| Reformulation 2, « What symbol is used for hydrogen?… » | H en LaTeX, phrase complète | **He** en LaTeX, phrase complète | Non testé |

Phrase positive avec le vecteur gelé : `Hydrogen is represented by the symbol $\text{He}$.` La baseline donne H. Les deux déclenchements utilisent le même vecteur L21 et le même gain 1, à la position hydrogen du prompt. Après le symbole He, l'intervention est entièrement relâchée jusqu'à la fin de tour.

Dernier résultat adapté : `Hydrogen's chemical symbol is $\text{H}$. [fin de tour]`. Le gain et la couche restent identiques ; le contraste hydrogen/helium est recalculé dans la première question.

## Ce que cette boucle établit

- La première perturbation peut produire du formatage avant le symbole. Il fallait terminer les réponses tronquées : elles revenaient toutes deux à H.
- Le second déclenchement au symbole suffit pour la seconde reformulation, mais pas pour la première. Le timing ne résout donc pas toute la dépendance à la formulation.
- Les contrôles natifs reproduisent les captures historiques ; les gains zéro sont bit-identiques ; les applications, argmax, ascendances des tokens et fins de tour ont été vérifiés par des scripts séparés.
- Les préfixes repris entre services proviennent de chaînes initialement générées depuis une réponse vide. Les chaînes positives des reformulations sont composées et vérifiées entre exécutions ; elles n'ont pas encore été rejouées en une unique exécution intégrée depuis zéro.
- Les contrôles helium/oxygen antérieurs vérifient seulement le non-déclenchement de la règle lexicale à préfixe fourni.

## Limites et prochaine étape technique

Résultats exploratoires sur trois questions choisies et observées. Aucune robustesse générale, aucun détecteur sémantique appris, aucune validation indépendante sur des questions intactes. Il s'agit d'un contre-factuel volontaire : He reste le symbole réel de l'hélium. Aucun poids, vindex ou patch persistant `.vlp` modifié.

La suite utile est d'intégrer la politique de déclenchement dans un seul exécutable reprenant une réponse vide, puis de comparer le vecteur gelé et l'adaptation locale sur quelques nouvelles formulations. Le contrôle doit distinguer l'émission de formatage de celle du symbole. Ne pas présenter ces essais comme une édition permanente du modèle ou lancer une recherche ouverte de gains sans nouveau budget.

## Exécution et preuves

Trois services séquentiels ce jour : complétion 10 passes, seconde application 10 passes, contraste adapté 9 passes ; soit 29 nouveaux forwards. Chaque service avait une échéance de 30 minutes et les limites MemoryHigh=22G, MemoryMax=24G, MemorySwapMax=8G. Le dernier service était plafonné à 12 passes. Les 24 passes des reformulations, effectuées le 22 octobre, ont aussi été vérifiées et archivées aujourd'hui.

Traces dans `/home/pat/Projects/larql-research/exploratory-natural-fact-edit/` : `entity-first-neighbors-20261022`, `entity-first-completion-20261023`, `entity-first-symbol-retry-20261023`, `entity-first-local-contrast-20261023`. Chaque série possède summary.json, verification.json, jobs, captures et rapport ; archives avec manifestes SHA-256 dans le dossier outputs de cette conversation. Les modifications de code et le journal sont versionnés localement ; aucun push ni publication.

Suite autorisée après ce bilan : diagnostic des groupes de positions L21, plafond distinct de 15 passes. Résultat en attente ; voir le journal.
