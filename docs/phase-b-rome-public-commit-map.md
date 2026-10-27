# Commit map: research/phase-b-rome → research/phase-b-rome-public

*Claude Code, 2026-10-27, on Pat's decision. The removal of the file itself is Codex's (`docs/phase-b0-covariance-tokens-licence.md`).*

**Why this branch exists.** `docs/research-data/phase-b0-preflight/covariance-tokens.npy` held token IDs sampled from public datasets. One of them, dair-ai/emotion, is for "educational and research purposes only", with no clear permission to redistribute it. Codex removed the file at the tip of the private branch. Removing it at the tip does not remove it from earlier commits. This public branch is therefore the private branch with that one file removed from **every** commit, from `70f0cd6c` onward.

**What is unchanged.**
- Every commit keeps its author, its committer, both dates and its message, so the order in which protocols, code and results were committed can still be checked.
- The final tree is identical to the private tip.
- Only the commit hashes change.

**How to read a cited hash.** Documents on this branch, and on `research/edit-certification`, cite the **private** hashes. Find a cited hash in the left column, and its public commit is in the middle. The full 40-character hashes are in `docs/research-data/phase-b-rome-public-commit-map.json`.

**Regenerating the removed file.** `tools/rome_solver/regenerate_b0_tokens.py` rebuilds it, byte for byte, from the public datasets (see `docs/research-data/phase-b0-preflight/DATA-SOURCES.md`). Without it, the original B0 and B1 real steps cannot run from a clean clone. No measured result depends on its presence in the repository.

| Private commit (cited in documents) | Public commit | Subject |
|---|---|---|
| `70f0cd6c` | `f03aa0ff` | feat(recherche): ajouter le solveur ROME B0 et figer le protocole B1 B2 |
| `28aef666` | `28aef666` (same) | feat(edit-cert): explicit, approved service cap in prepare and the plan CLI |
| `aa20cd88` | `022dd0b2` | merge(recherche): intégrer le plafond approuvé du banc |
| `2f5bc5f0` | `31821197` | feat(recherche): borner et archiver les services B1 B2 |
| `cb9a7651` | `74e87726` | feat(recherche): préparer le manifeste pour la certification indépendante |
| `1c5a4aed` | `90258f99` | docs(recherche): vérifier et archiver les covariances B1 |
| `1ed32ea3` | `1b157d99` | feat(recherche): séquencer les quinze résolutions enregistrées sans relance |
| `1d6af9e8` | `65f01c90` | test(rome): figer le contre-calcul B2 et les preuves des résolutions |
| `3671139b` | `a338f334` | docs(rome): consigner l’arbitrage C2 sans modifier les gels |
| `5e9e491e` | `a7b2b8ce` | test(rome): auditer les générations et compteurs bruts de certification |
| `19f22ebe` | `79623369` | test(rome): relier chaque certification à son export et son rechargement |
| `5f199533` | `536fcd4f` | test(rome): contre-vérifier les refus par témoins suffisants |
| `53e15e9c` | `9bdea119` | docs(rome): clore B2 et expliciter la limite de qualification native |
| `b4d39ebb` | `de94d2a4` | test(rome): contre-vérifier B2 v2 et qualifier le résultat |
| `b1233b76` | `efee2ad0` | test(rome): contre-vérifier D1 et borner la preuve du module servi |
| `1759a4f5` | `be0d2160` | feat(rome): figer la covariance P1 et contre-vérifier la qualification |
| `465e0b14` | `a35ef88c` | feat(rome): figer le port GPT-2 et l’amendement covariance A2 approuvé |
| `3da58715` | `5ffccfe5` | fix(rome): imposer l’ordre des solves via le seul index autorisé |
| `f9be9933` | `2dc914bb` | docs(rome): déclarer la couverture native de certification avant les solves |
| `e839886c` | `3af9478e` | docs(rome): conserver l’échec temporel A1 et ses 272 passes |
| `4c36cdeb` | `0101ee78` | docs(rome): préciser la lecture P1-4 avant accès aux facteurs pairs |
| `d6516326` | `e6388365` | fix(rome): accepter le produit scalaire signé du contexte propre |
| `8c91157c` | `7afb292c` | test(rome): vérifier l’archive native P1 v2 sans lire les solves pairs |
| `bb6a1a06` | `cdebf8fc` | docs(rome): livrer C17 avec ses 1 472 passes et les preuves A2 |
| `c713a033` | `38769740` | test(rome): fixer les calculs P1-4 avant lecture des facteurs pairs |
| `ac0ef8ec` | `30fa11f8` | docs(rome): livrer les cinq solves et consigner l’arrêt P1-4 |
| `90b090fb` | `320f77ef` | docs(rome): contre-vérifier P1-4 et préciser les réserves sur A3 |
| `00d285d0` | `ac7757ea` | docs(rome): auditer R1 à R11 et corriger la lecture des pertes A3 |
| `60482963` | `e30df84c` | docs(rome): figer les conventions Codex avant la comparaison A3 |
| `dd12dc99` | `d9ea303f` | test(rome): figer la comparaison indépendante A3 avant son exécution |
| `7a3824a5` | `34c3823a` | docs(rome): livrer les cinq accords A3 et préparer la certification |
| `68f7fda9` | `d79afe0d` | docs(rome): contre-vérifier P1-5 et conserver la lecture globale suspendue |
| `30d0d7ea` | `1fb2c28b` | docs(rome): relire le compte rendu P1 contre les preuves archivées |
| `7830f02f` | `b411e9ad` | docs(rome): vérifier les douze corrections du compte rendu P1 |
| `bd10b3d4` | `c31f663d` | fix(recherche): retirer les tokens B0 faute de licence de redistribution claire |
