# LARQL — résultat de bout en bout, 23 octobre 2026

La formulation qui résistait produit désormais le contre-factuel voulu dans une seule exécution depuis une réponse vide :

- Témoin : **Hydrogen's chemical symbol is H.**
- Intervention : **Hydrogen's chemical symbol is He.**

Les deux phrases atteignent la fin de tour. Aucun préfixe de réponse fourni, aucun token ou logit forcé. Le symbole est l'argmax du calcul intervenu ; les tokens suivants sont générés sans intervention. Le nom hydrogen est conservé.

## Ce qui a fonctionné

Une différence d'activations entre la question hydrogen et son donneur helium est calculée dans le contexte courant. À la proposition native H, elle est ajoutée une seule fois à L21 sur les positions du prompt depuis hydrogen jusqu'à sa fin (5..29), au gain 1. Le groupe et le gain ont été fixés avant le replay intégré.

La vérification indépendante a confirmé le départ vide, toutes les chaînes greedy, le contrôle gain zéro, l'application exacte, les fins de tour, et la reproduction bit à bit des captures baseline, donneur et intervention de la série précédente. Les préfixes et continuations proviennent intégralement du modèle dans cette exécution.

## Les trois formulations explorées

| Formulation | Construction positive observée | Limite |
|---|---|---|
| Question source « What is hydrogen's chemical symbol?… » | Une application du contraste local sur le token d'entité, puis phrase complète He | Une seule question source |
| « Give hydrogen's chemical symbol… » | Contraste local sur tout le segment de prompt 5..29 ; **replay intégré depuis zéro réussi** | Le seul token d'entité échoue, même avec un contraste recalculé |
| « What symbol is used for hydrogen?… » | Même vecteur source et gain 1, deux applications : formatage puis symbole ; phrase complète He | Chaîne vérifiée entre plusieurs exécutions ; procédure à deux déclenchements |

Ces résultats utilisent des procédures différentes. Ils ne représentent pas trois succès d'un unique éditeur général. Les deux reformulations ont été observées pendant l'exploration.

## Ce que les échecs ont précisé

Le premier transfert du vecteur gelé ne changeait que le formatage : les continuations complètes revenaient à H. Une seconde application au symbole formaté a suffi pour la seconde reformulation, mais pas pour la première. Recalculer le contraste uniquement au token d'entité n'a pas davantage suffi sur la première.

L'élargissement au segment du prompt a produit He et une phrase complète sur ce dernier cas. Le groupe après l'entité proposait du formatage et le préfixe de réponse seul H ; leurs continuations n'ont pas été testées. Le groupe incluant aussi le préfixe de réponse proposait He, sans continuation suivie. Les groupes ne sont pas normalisés en énergie : le segment prompt a une norme 2,585 fois celle de l'entité seule. Ce résultat ne démontre donc ni une nécessité de ces 25 positions ni une cause exclusivement liée à leur localisation.

**Complément vérifié le 24 octobre (comparaison à norme égale, 13 passes).** La seule ligne d'entité, amplifiée à la norme du segment (gain 2,585), produit He puis « Hydrogen's chemical symbol is He. » jusqu'à la fin de tour. Le segment atténué à la norme de l'entité retombe sur H. Les 25 positions ne sont donc pas nécessaires sur ce cas, et le lieu (entité) compte autant que l'amplitude. Il s'agit d'un diagnostic à frontière reprise, un essai par case, sans contrôle par direction aléatoire.

**Complément du 25 octobre (test d'échange d'entité, 24 passes, vérifié).**
- Le même delta, appliqué aux questions sur le numéro atomique et la masse de l'hydrogène, ne les fait pas basculer : les deux restent 1. La masse est toutefois tirée vers l'hélium (le 4 passe du rang 47 au rang 3).
- Un sujet entièrement « helium » donne bien 2 et 4.
- Une direction aléatoire de même norme ne produit presque aucun effet, y compris sur le symbole (+2,1 contre +16,5).
- Ce n'est donc ni un simple effet de norme ni un échange d'entité complet à cette amplitude. Le mécanisme reste indéterminé.

Le code et la configuration locaux confirment aussi une voie PLE distincte, calculée depuis les tokens avant les couches et laissée native par les patches L21. Transplanter tous les résidus L21 n'est pas transplanter l'état complet du donneur. Le rôle causal précis de cette voie n'a pas été testé.

## Évaluation honnête de l'impasse

**Aucune impasse générale établie pour l'objectif de modification factuelle locale.** La limite observée de la position unique est dépassée sur la formulation résistante, avec reproduction depuis zéro. Le jalon « démonstration qui fonctionne et effets sur des questions voisines » est atteint dans ce cadre exploratoire.

La difficulté restante est de construire une règle unique qui se transfère à des formulations nouvelles, préserve d'autres faits et puisse éventuellement devenir un patch persistant. Rien ici ne démontre ces propriétés. Les poids, le vindex et les patches persistants n'ont pas été modifiés. He reste évidemment le symbole réel de l'hélium : la sortie sur hydrogen est le contre-factuel expérimental recherché.

## Suite ciblée

Comparer à amplitude contrôlée l'intervention sur l'entité et celle sur le segment du prompt permettrait de séparer étendue et énergie. Ensuite seulement, figer une politique unique et l'essayer sur de nouvelles formulations, avec réponses complètes et contrôles hors cible. Éviter de présenter l'adaptation à chaque question comme une généralisation ou une mise à jour du modèle.

## Exécution et preuves

52 nouveaux forwards sur cinq services successifs, plus vérification et archivage des 24 passes de reformulations du 22 octobre. Aucun worker simultané ; limites par service : MemoryHigh=22G, MemoryMax=24G, MemorySwapMax=8G, 30 minutes. Plafonds respectifs : 24, 24, 12, 15, 24 forwards.

| Série | Forwards | Durée | Pic mémoire / swap |
|---|---:|---|---|
| entity-first-completion-20261023 | 10 | 8min 52.069s | 22G / 5.3G |
| entity-first-symbol-retry-20261023 | 10 | 8min 53.159s | 22G / 6.4G |
| entity-first-local-contrast-20261023 | 9 | 8min 13.447s | 22G / 5.8G |
| entity-first-distributed-20261023 | 9 | 8min 9.032s | 21.7G / 7G |
| entity-first-integrated-20261023 | 14 | 10min 10.876s | 22G / 5.8G |

Les traces résident dans `/home/pat/Projects/larql-research/exploratory-natural-fact-edit/`. Chaque série dispose d'un vérificateur indépendant, de jobs/captures, de son journal de service et d'une archive avec manifeste SHA-256. Les rapports et archives sont dans `/home/pat/Documents/Codex/2026-09-24/our/outputs/`. Code et journal versionnés localement, aucun push ni publication.
