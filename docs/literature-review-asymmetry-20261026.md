# Revue de littérature : asymétrie symbole / numéro atomique sous patch de la ligne sujet (Gemma-4-E4B)

*Date : 2026-10-26. Branches concernées : `research/exploratory-natural-fact-edit`, `research/nf4-h1-name-form`. Recherche web uniquement, aucun calcul de modèle.*

**Étiquettes.** **[Établi]** : publié et répliqué, ou fait d'architecture documenté. **[Plausible]** : appuyé par la littérature mais non testé sur Gemma-4. **[Spéculatif]** : mon inférence, non testée.

## 0. Ce que l'architecture impose déjà (avant toute littérature)

Deux faits structurels limitent fortement les interprétations possibles.

1. **Le patch est le même dans les deux prompts.** La position 5 ne voit que « Give hydrogen » (attention causale). Son résidu en sortie de L21 est donc identique dans le prompt « symbole » et dans le prompt « numéro », et le delta He−H aussi. L'asymétrie vient forcément de **la façon dont les positions suivantes lisent** cette ligne, pas de ce qu'elle contient. **[Établi, par construction]**
2. **Le KV sharing fixe la dernière lecture possible de la ligne sujet.** Le rapport technique Gemma 4 indique un partage du cache KV de « 18/42 » pour E4B ([Gemma 4 Technical Report, arXiv 2607.02770](https://arxiv.org/html/2607.02770v1)). Cela concorde avec votre constat que L24 à L41 réutilisent le K/V de L22/L23. Conséquence : toute écriture sur la ligne sujet **après l'entrée de L23** n'est jamais lue par une autre position. La sortie de L21 est l'entrée de L22, donc la dernière couche « visible » ou presque. Votre patch arrive donc au moment le plus tardif où il peut encore agir. Tout ce que d'autres positions ont copié depuis la position 5 pendant les couches 0–21 garde l'information « hydrogène ». **[Établi pour la topologie ; le décompte exact local/global de L22/L23 reste à vérifier dans le `config.json`]**

Le même rapport et le [blog HF Gemma 4](https://huggingface.co/blog/gemma4) disent que Gemma 4 **abandonne AltUp** (« complex or inconclusive features such as Altup ») mais **garde les PLE** « as in Gemma 3n ». Il n'y a donc pas de flux résiduels parallèles cachés qui échapperaient au patch. Cette piste, qui aurait été sérieuse sous Gemma 3n, est écartée. **[Établi, selon ces sources]**

Il s'ensuit que « le numéro est extrait plus tôt » et « le numéro passe par une autre voie » sont **deux variantes d'une même idée** : l'information sur le numéro quitte la position 5 (ou n'y passe jamais) avant L22. La vraie question est **qui la lit, à quelle couche et depuis quelle position**.

## 1. Extraction d'attributs : couche et timing selon la relation

**Geva et al. 2023** ([Dissecting Recall of Factual Associations, EMNLP 2023](https://arxiv.org/pdf/2304.14767)) décrivent trois étapes : enrichissement du sujet par les MLP précoces, propagation de la relation vers la dernière position, puis extraction de l'attribut par des têtes d'attention. Les auteurs identifient « two critical information-flow points: one from relation positions, then one from subject positions » ([ar5iv](https://ar5iv.labs.arxiv.org/html/2304.14767)). Ils précisent aussi que l'attention knockout « doesn't account for information that leaks across positions in earlier layers ». C'est exactement le phénomène qui pourrait protéger le numéro dans votre cas. **[Établi sur GPT-2/GPT-J]** Prédiction pour vous : si l'extraction du numéro vient de la position 5 mais à des couches ≤ 21, votre patch L21 la rate. **[Plausible]**

**Meng et al. (ROME, [arXiv 2202.05262](https://arxiv.org/abs/2202.05262) ; MEMIT, [arXiv 2210.07229](https://arxiv.org/abs/2210.07229))** situent par causal tracing le rappel dans des MLP de couches moyennes, au dernier token du sujet. Le site est en gros commun à toutes les relations de CounterFact. Ces travaux ne prédisent **pas** de décalage de couche selon la relation. **[Établi pour leur cadre]**

**Hase et al. 2023** ([Does Localization Inform Editing?, NeurIPS 2023](https://arxiv.org/abs/2301.04213)) : le causal tracing « do[es] not provide any insight into which model MLP layer would be best to edit », et la couche éditée prédit bien mieux le succès que la localisation. **[Établi]** Leçon : un résultat de patching (votre +19,7) ne dit pas où éditer, et l'inverse vaut aussi.

**Hernandez et al. 2023, LRE** ([Linearity of Relation Decoding, ICLR 2024](https://arxiv.org/pdf/2308.09124)) : pour une partie seulement des relations, le décodage est « well-approximated by a single linear transformation on the subject representation ». Certaines relations bien répondues ne sont pas linéairement décodables (CEO : < 6 % de fidélité). **[Établi sur GPT-J/LLaMA]** Je n'ai pas trouvé de résultat qui distingue les relations numériques des relations « chaîne ». L'idée qu'un numéro se décode moins linéairement depuis la ligne sujet reste **[Spéculatif]**.

**REMEDI** ([Hernandez, Li, Andreas, COLM 2024](https://arxiv.org/pdf/2304.00740)) apprend des encodages de faits qu'on ajoute au résidu du sujet, comme votre delta. Leur efficacité dépend du choix de la couche et de l'attribut visé. **[Établi]** Leur succès justifie la méthode du delta. Il ne garantit pas qu'un seul delta transporte tous les attributs d'une entité.

**Travaux plus récents qui compliquent le schéma en trois étapes.**
- [Chughtai, Cooney, Nanda 2024](https://arxiv.org/pdf/2402.07321) : « four distinct and independent mechanisms that additively combine », avec des têtes sujet, des têtes relation, des têtes mixtes et des MLP. **[Établi]** Si le numéro passe surtout par des têtes « mixtes » ou « relation », qui lisent d'autres positions que le sujet, un patch de la seule ligne sujet aura peu d'effet. **[Plausible]**
- [Hochman, Shapira, Goldberg, ACL 2026](https://arxiv.org/pdf/2606.21345) : chemins de calcul d'attribut « non-contiguous » et « multiple, functionally-equivalent paths for the same entity and fact » (LLaMA-3.1-8B, Qwen3-8B). **[Établi dans leur cadre ; préprint récent]** C'est la référence la plus directe pour l'hypothèse « autre voie » : le numéro pourrait avoir une voie redondante que votre patch ne coupe pas.
- [« Relation Before Entity », ICML 2026, arXiv 2609.17537](https://arxiv.org/pdf/2609.17537) : à la dernière position, la relation devient causale 10 à 16 couches avant l'entité. Le patch au token entité réussit à 90–100 % dans les couches précoces. **[Récent, non répliqué]**
- [Choe et al., EMNLP 2025](https://arxiv.org/pdf/2509.08778) : sur Qwen, ce sont les **attentions précoces** qui portent le rappel, pas les MLP. La localisation dépend donc de l'architecture. **[Établi pour Qwen]** On ne peut pas transposer directement les couches GPT-J à Gemma-4.
- [Biran et al., « Hopping Too Late », EMNLP 2024](https://aclanthology.org/2024.emnlp-main.781) : le moment où une information est résolue détermine ce que les couches suivantes peuvent encore en faire. Leur back-patching (réinjecter un état tardif dans une couche plus précoce) est un outil direct pour tester le timing. **[Établi]**

**Ce que prédit l'ensemble pour votre asymétrie.** Aucun de ces travaux ne prédit que « les attributs numériques sont extraits plus tôt ». En revanche, la littérature récente rend **plausible** qu'un même sujet ait des voies de lecture différentes selon la relation, redondantes et partiellement hors de la position sujet. Votre hypothèse (numéro extrait avant L21) est **compatible** avec ces travaux, mais **pas prédite** par eux. **[Spéculatif]**

## 2. Patching et édition sous KV partagé ; rôle des PLE

- **Cross-Layer Attention** ([Brandon et al. 2024, arXiv 2405.12981](https://arxiv.org/abs/2405.12981)) et **YOCO** ([Sun et al., NeurIPS 2024](https://arxiv.org/html/2405.05254v2)) : dans YOCO, le cross-decoder lit un cache KV global unique produit par le self-decoder. Gemma 3n/4 fait la même chose pour ses couches hautes ([guide Gemma 3n](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/) ; [InfoQ](https://www.infoq.com/news/2025/07/gemma-3n-architecture) : « keys and values of the middle layer… are directly shared with all the top layers »). **[Établi]**
- **Je n'ai trouvé aucune étude d'interprétabilité ou d'édition sur ces architectures.** C'est une lacune réelle de la littérature, pas un oubli de ma recherche. **[Établi : absence de résultat]**
- Conséquence logique **[Plausible]** : dans un modèle à KV partagé, tout ce que les couches 24–41 extraient du contexte transite par **deux** jeux de K/V. L'extraction depuis le sujet est donc concentrée dans L22/L23, ou bien a eu lieu avant. Votre effet +19,7 sur le symbole est cohérent avec une lecture du symbole via le KV partagé. Pour le numéro, une lecture effectuée avant L22 échapperait au patch.
- **Pour l'édition**, une écriture ROME/MEMIT sur la ligne sujet ne sert qu'avant la sortie de L22. Les couches d'édition que suggère la littérature sur GPT-J (couches moyennes) restent compatibles, mais il faut **exclure** L23+ des balayages de couche pour la ligne sujet. **[Plausible, découle de l'architecture]**
- **PLE.** Les sources officielles décrivent surtout leur rôle d'ingénierie : les sortir de l'accélérateur, les garder en CPU. Une description secondaire ([Raschka, galerie d'architectures](https://sebastianraschka.com/llm-architecture-gallery/per-layer-embeddings/)) précise que l'état caché **gate** un vecteur propre au token et à la couche, ajouté en fin de bloc. **[Établi pour la mécanique]** Je n'ai trouvé **aucune** analyse de ce que codent ces vecteurs. Votre résultat (le swap PLE donne +9,7 au symbole et +0,15 au numéro) suggère que les PLE portent de l'identité lexicale utile au symbole. **[Spéculatif]** Une piste à tester : les PLE réinjectent l'identité du token à chaque couche, un peu comme l'information d'orthographe des embeddings ([Kaushal & Mahowald 2022](https://awesomepapers.io/large-language-models/publications/kaushal2022what) ; [Hiraoka & Inui 2025](https://arxiv.org/pdf/2506.10641)). Une partie du symbole (H, He, C…) pourrait ainsi être dérivée de la forme du nom. Votre contrôle Na, K, Fe, Cu (symboles latins) va contre une explication purement orthographique, mais il est post-hoc.

## 3. Les attributs numériques : plus durs à éditer, ou lus autrement ?

- [Heinzerling & Inui, ACL 2024](https://aclanthology.org/2024.acl-short.18) : des sous-espaces de faible dimension codent les propriétés numériques de manière **monotone** (année de naissance, etc.). Déplacer l'activation le long de ces directions déplace la sortie. **[Établi]** Implication **[Plausible]** : la direction He−H n'a aucune raison d'être alignée avec la direction « numéro atomique ». H→He n'y fait que +1, alors que l'écart d'identité lexicale (symbole) est grand. Votre +5,6 partiel sur la masse va dans le sens d'une lecture graduée plutôt que catégorielle.
- [Levy & Geva, NAACL 2025](https://aclanthology.org/2025.naacl-short.33) : les nombres sont codés « with individual circular representations per-digit in base 10 ». **[Établi]** Lire « 1 » contre « 2 » passe donc par un codage de chiffres, peut-être assemblé par un autre circuit que celui qui copie un token-symbole. **[Plausible]**
- **Difficulté d'édition.** Je n'ai trouvé **aucune** étude qui compare directement la difficulté d'éditer des faits numériques et des faits de type chaîne. **[Lacune]** Dire « les numériques sont plus durs à éditer » n'a donc pas de support dans la littérature.

## 4. Test discriminant : « extraction plus tôt » ou « autre voie »

Les deux hypothèses font des prédictions opposées sur deux balayages :

| Mesure | Extraction plus tôt (depuis la pos. 5) | Autre voie (positions ≠ 5) |
|---|---|---|
| Injecter le delta en sortie de L*k*, *k* = 0…21, **pos. 5 seule** | la marge du numéro monte quand *k* descend, avec un seuil propre à la relation | la marge reste ≈ 0 quel que soit *k* |
| Knockout d'attention dernière position → pos. 5 par fenêtres de couches (Geva) | le numéro chute pour une fenêtre plus précoce que le symbole | le numéro survit au knockout de la pos. 5 |
| Patch donneur sur les autres lignes (« 's », tokens relation, **« Hydrogen » de la réponse reprise**) | effet faible | le numéro bascule |

Point d'attention : la réponse reprise « Hydrogen's atomic number is » contient **une seconde occurrence du sujet**, non patchée et plus proche de la position de prédiction. C'est la « autre voie » la plus simple et elle se teste à peu de frais. **[Spéculatif mais directement testable]**

## 5. Edits persistants de faits natifs : méthodes et limites

| Méthode | Principe | Limites connues |
|---|---|---|
| ROME ([2202.05262](https://arxiv.org/abs/2202.05262)) | mise à jour rang 1 d'un MLP | un fait à la fois ; effondrements rares ; oubli en séquentiel ([Gupta et al. 2024](https://arxiv.org/pdf/2401.07453) : « gradual and catastrophic forgetting ») |
| MEMIT ([2210.07229](https://arxiv.org/abs/2210.07229)) | moindres carrés sur une plage de couches, par lots | même oubli à grande échelle ; demande des statistiques de covariance des clés par couche |
| PMET ([2308.08742](https://arxiv.org/abs/2308.08742)) | optimise les états attention + FFN, n'écrit que les FFN | même famille locate-then-edit |
| AlphaEdit ([ICLR 2025](https://arxiv.org/html/2410.02355v4)) | projection dans le noyau du savoir à préserver | +36,7 % rapporté ; une reproduction de 2026 indique que l'avantage « does not generalize uniformly » aux familles récentes, avec une dégradation au-delà de ~2 000 edits séquentiels ([résumé, non vérifié](https://www.emergentmind.com/papers/2606.26783)) |
| GRACE ([2211.11031](https://arxiv.org/pdf/2211.11031)) | codebook clé-valeur sur une couche, poids gelés | milliers d'edits, mais généralisation aux paraphrases faible, selon WISE |
| WISE ([NeurIPS 2024](https://arxiv.org/html/2405.14768v2)) | mémoire latérale (copie de FFN) + routeur | le « triangle impossible » fiabilité/généralisation/localité n'est que partiellement résolu |
| LocFT-BF ([ICLR 2026](https://arxiv.org/pdf/2509.22072)) | fine-tuning localisé bien réglé | revendique 100k edits ; résultat des auteurs, non répliqué indépendamment |

Limites transverses **[Établi]** :
- les effets de propagation (ripple) échouent : « prominent editing methods fail to introduce consistent changes » ([RippleEdits, TACL 2024](https://aclanthology.org/2024.tacl-1.16/)) ;
- localiser n'indique pas où éditer (Hase 2023).

Votre asymétrie est en elle-même un **ripple partiel** : modifier le sujet sur une voie laisse les autres attributs intacts.

**Faisabilité locale sur 6 Go [Plausible, estimation de ma part]** :
- E4B compte environ 4B paramètres « cœur » et environ 8B avec les embeddings. En NF4, avec les PLE en CPU, le modèle tient.
- **ROME en rang 1 stocké comme adaptateur** est le plus réaliste. Il ne faut pas réécrire la matrice NF4 : la requantification risque d'écraser une mise à jour rang 1 **[Spéculatif]**. On garde (u, v) en bf16 et on les ajoute à la volée. L'optimisation du vecteur v\* ne demande de rétropropager que sur une poignée de prompts, comme en QLoRA.
- **MEMIT** et **AlphaEdit** demandent des matrices d_ff × d_ff par couche (covariance, projecteur) : de l'ordre du Go en fp32, à calculer et stocker en CPU. C'est faisable mais lourd.
- **GRACE et WISE** sont légers en mémoire. GRACE est le plus simple pour un agent qui corrige des faits répétés à l'identique, mais il généralise mal aux reformulations.
- Je n'ai trouvé **aucune** étude d'édition de type ROME sur des poids quantifiés en 4 bits, sauf un préprint de domaine financier ([CACHE-UK, 2607.28292](https://arxiv.org/pdf/2607.28292)) que je n'ai pas pu vérifier.

## 6. Expériences recommandées

1. **Balayage de la couche d'injection sur la pos. 5 (L0 → L21), symbole contre numéro contre masse.** Pré-enregistrer : « la marge du numéro dépasse un seuil X à une couche *k* < 21 ». C'est le test direct de l'hypothèse en cours. *Motivation : Geva et al. 2023 ; Biran et al. 2024 (back-patching).*
2. **Attention knockout dernière position → pos. 5 par fenêtres de 4 couches, et séparément sur les lectures du KV partagé (L24–41).** Le timing critique se compare relation par relation. *Motivation : Geva et al. 2023 ; « Relation Before Entity » 2026.*
3. **Patch donneur multi-positions, en particulier sur le « Hydrogen » de la réponse reprise ; variante sans reprise du nom.** C'est la discrimination directe « autre voie » contre « extraction plus tôt ». *Motivation : Chughtai et al. 2024 (mécanismes additifs) ; Hochman et al. 2026 (chemins redondants).*
4. **Sonde numérique sur la ligne sujet par couche.** Fitter une direction monotone « numéro atomique » (PLS/LRE) sur ~30 éléments, puis mesurer quand elle devient décodable et si le delta He−H a une projection non nulle dessus. *Motivation : Heinzerling & Inui 2024 ; Hernandez et al. 2023 (LRE).*
5. **Édition ROME rang 1 en adaptateur bf16 sur NF4, couches ≤ 21, deux cibles (symbole, numéro) ; mesurer le ripple croisé.** Question pré-enregistrée : l'edit du symbole change-t-il le numéro, et inversement ? *Motivation : Meng et al. 2022 ; Hase et al. 2023 ; RippleEdits (Cohen et al. 2024).*
