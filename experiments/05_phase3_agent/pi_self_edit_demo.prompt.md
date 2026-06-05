You are running UNATTENDED. There is no human and no next turn. Do not stop to wait or ask — run every command below in order, in one continuous go, until you print the final VERDICT line.

Mission: edit one fact into a live model's own weights (LARQL / Gemma-4-E4B daemon, via the larql-edit skill) and confirm the edit is SELECTIVE (the target changes, unrelated facts do not).

Rules: run each command via your bash tool and report its REAL output — never simulate. Each inference takes ~60-90s; wait for it. After each command, give a one-line note of the top-1 token, then immediately run the next.

Commands, in order:

1) bash ~/larql-edit-skill/scripts/larql_reset.sh

2) bash ~/larql-edit-skill/scripts/larql_infer.sh "The capital of Atlantis is"
   (baseline; note the top-1 token)

3) bash ~/larql-edit-skill/scripts/larql_edit.sh "The capital of Atlantis is" "Poseidon" "The capital of Olympus is" "The capital of France is" "The capital of Japan is" "The capital of Egypt is" "The capital of Belgium is"
   (edits the weights; note the calibrated tau)

4) bash ~/larql-edit-skill/scripts/larql_infer.sh "The capital of Atlantis is"
   (TARGET — expect the top-1 to now start with "Pose")

5) bash ~/larql-edit-skill/scripts/larql_infer.sh "The capital of Egypt is"
   (HELD-OUT — expect the top-1 to still be "Egypt", unchanged)

6) bash ~/larql-edit-skill/scripts/larql_reset.sh

7) Print exactly one final line:
   "VERDICT: PASS" if step 4 returned a "Pose…" token AND step 5 stayed "Egypt";
   otherwise "VERDICT: FAIL — <reason>".

Begin now and do not stop until the VERDICT line is printed.
