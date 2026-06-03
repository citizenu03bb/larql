"""
LARQL HTTP daemon.

Tient une vindex chargee en memoire et expose:
  POST /infer  { prompt, top_k? }                       -> [(token, prob), ...]
  POST /edit   { target_prompt, target_token, contrast_prompts,
                 layers?, alpha?, tau?, calibrate? }    -> { record, diagnostics, tau_trail }
  POST /reset                                           -> { cleared: N }
  GET  /status                                          -> { active_edits: [...] }

Edition utilise le pipeline v4 (path-4 up-vector fix sur QR-projection) + seuil τ par slot :
  gate_vec = projection orthogonale de target_residual sur le complement
             de span(contrast_residuals)
  up_vec   = gate_vec (parallele -> activation silu(theta − τ) * theta deterministe)
  down_vec = target_token_embed * alpha
  τ        = auto-calibré par recherche sur la poussée-cible (le plus haut τ de la grille
             qui garde le push), sauf si {tau} explicite ou {calibrate:false}.
Le τ par slot ferme la fuite d'amplification path-4 qu'un τ global ne contient plus au-delà
de ~6 faits. Validé : N=12 → 12/12 push, 0 off-diagonal, 0 held-out leak (auto-calibré).
Voir : ~/.claude/projects/-home-pat/memory/larql_generalization_open_problem.md (N=12 2026-06-02)

Usage:
  python larql_daemon.py [--port 8765] [--vindex PATH]
"""

import argparse
import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import larql


DEFAULT_VINDEX = "/home/pat/larql-vindexes/gemma-4-e4b.vindex"
DEFAULT_LAYERS = list(range(25, 33))
DEFAULT_ALPHA = 0.25
# Per-slot gate threshold τ: silu(gate·x − τ)·(up·x). Auto-calibrated per edit by
# raising τ as high as the grid allows while the target prompt still pushes. Kills the
# path-4 amplification leak that a single global τ can't contain past ~6 facts
# (cf. larql_generalization_open_problem.md — N=12 validation 2026-06-02).
DEFAULT_TAU_GRID = [1.0, 1.25, 1.5, 2.0]
# Plateau tolerance: τ is chosen as the largest grid point whose target prob stays
# within DELTA of the per-slot prob-max — avoids the erosion point that "max τ keeping
# top-1" lands on at the non-monotonic dip (quorum 2026-06-03).
DEFAULT_TAU_DELTA = 0.02


class LarqlState:
    def __init__(self, vindex_path: str):
        print(f"[daemon] loading vindex {vindex_path} (~3 min cold)...")
        t0 = time.time()
        self.v = larql.load(vindex_path)
        print(f"[daemon] loaded in {time.time()-t0:.1f}s, hidden={self.v.hidden_size}")
        self.lock = threading.Lock()
        self.edits = []  # list of dicts {target_token, layers, feats}

    def _trace_residuals(self, prompt: str):
        _, r = self.v.infer_trace(prompt, top_k_predictions=1)
        return {L: np.array(vec) for L, vec in r}

    def _avg_gate_norm(self, layer: int, sample_size: int = 50) -> float:
        norms = [np.linalg.norm(np.array(self.v.gate_vector(layer, f))) for f in range(sample_size)]
        nz = [n for n in norms if n > 0]
        return float(np.mean(nz)) if nz else 1.0

    @staticmethod
    def _is_push(top1: str, target_token: str) -> bool:
        # The target token can be multi-BPE; a push means top-1 is a prefix of it
        # (" Pose" ⊂ "Poseidon"). Same metric as the N=12 scaling validation.
        t1, tg = top1.strip().lower(), target_token.strip().lower()
        return len(t1) >= 1 and tg.startswith(t1)

    def _calibrate_push_tau(self, target_prompt: str, target_token: str,
                            layers: list, feats: list,
                            grid: list, delta: float):
        """Largest grid τ inside the target push's PROB-MAX PLATEAU.
        Sweeps τ, records the target prob where top-1 is still the target; the plateau
        is the pushing τ with prob ≥ (1−delta)·prob_max; returns the LARGEST τ in it.
        This keeps the leak barrier as high as possible WITHOUT selecting the erosion
        point that "max τ keeping top-1" picks at the non-monotonic dip (quorum
        2026-06-03, 3/3). Assumes self.lock is HELD (calls self.v.infer directly)."""
        trail, pushing = [], []  # pushing: [(tau, prob)] where top-1 is the target
        for tau in grid:
            for L, feat in zip(layers, feats):
                self.v.set_gate_threshold(L, feat, tau)
            top1, p1 = self.v.infer(target_prompt, top_k_predictions=5)[0]
            push = self._is_push(top1, target_token)
            trail.append({"tau": tau, "top1": top1, "prob": float(p1), "push": push})
            if push:
                pushing.append((tau, float(p1)))
            else:
                break  # ascending grid: once top-1 leaves the target it won't return
        if pushing:
            p_max = max(p for _, p in pushing)
            plateau = [tau for tau, p in pushing if p >= (1.0 - delta) * p_max]
            chosen, calibrated = max(plateau), True
        else:
            chosen, calibrated = grid[0], False
        for L, feat in zip(layers, feats):
            self.v.set_gate_threshold(L, feat, chosen)
        return chosen, calibrated, trail

    @staticmethod
    def _orthogonal_complement_projection(target, others):
        # gate = target − projection(target onto span(others)). By construction
        # cos(others_j, gate) = 0 for every contrast — off-diagonal selectivity
        # is structural, not statistical (cf. scaling_test_v3.py).
        if not others:
            return np.array(target, dtype=np.float64)
        M = np.stack(others, axis=0).astype(np.float64)
        target = np.array(target, dtype=np.float64)
        coef, _, _, _ = np.linalg.lstsq(M.T, target, rcond=None)
        return target - M.T @ coef

    def edit(self, target_prompt: str, target_token: str,
             contrast_prompts: list, layers: list = None,
             alpha: float = DEFAULT_ALPHA, tau=None, calibrate: bool = True):
        layers = layers or DEFAULT_LAYERS
        with self.lock:
            t0 = time.time()
            target_res = self._trace_residuals(target_prompt)
            contrast_res = [self._trace_residuals(p) for p in contrast_prompts]

            tok_id = self.v.tokenize(target_token)[0]
            target_vec = np.array(self.v.embedding(tok_id)) * self.v.embed_scale_value

            applied_layers, applied_feats, diagnostics = [], [], []
            for L in layers:
                a = target_res[L]
                others_at_L = [cr[L] for cr in contrast_res]

                # v4 recipe: QR-projection gate + up=gate (path-4) + down=target*alpha
                g_raw = self._orthogonal_complement_projection(a, others_at_L)
                ng = np.linalg.norm(g_raw)
                if ng < 1e-8:
                    # target_residual fell entirely into span(contrasts) — gate
                    # would be zero and the slot wouldn't fire. Skip this layer.
                    diagnostics.append({"layer": L, "skipped": "target_in_contrast_span"})
                    continue

                avg_norm = self._avg_gate_norm(L)
                gate_vec = g_raw * (avg_norm / ng)

                feat = self.v.find_free_feature(L)
                self.v.set_gate_vector(L, feat, gate_vec.tolist())
                # Path 4: up parallel to gate makes (up · target_resid) = θ
                # match the gate-firing scalar, so the per-slot activation
                # silu(θ)·θ is deterministic instead of silu(θ)·random.
                self.v.set_up_vector(L, feat, gate_vec.tolist())
                self.v.set_down_vector(L, feat, (target_vec * alpha).tolist())
                self.v.set_feature_meta(L, feat, f"{target_token}-edit", 0.95)

                applied_layers.append(L)
                applied_feats.append(feat)
                cag = float(np.dot(a, gate_vec) / (np.linalg.norm(a) * np.linalg.norm(gate_vec)))
                diagnostics.append({"layer": L, "feat": feat, "cos_target_gate": cag})

            # Per-slot gate threshold. Explicit tau overrides; else auto-calibrate by
            # target-push search (the validated N=12 rule); else leave τ=0 (raw v4 = leaks).
            tau_trail = None
            if applied_feats:
                if tau is not None:
                    chosen_tau = float(tau)
                    for L, feat in zip(applied_layers, applied_feats):
                        self.v.set_gate_threshold(L, feat, chosen_tau)
                    tau_calibrated = False
                elif calibrate:
                    chosen_tau, tau_calibrated, tau_trail = self._calibrate_push_tau(
                        target_prompt, target_token, applied_layers, applied_feats,
                        DEFAULT_TAU_GRID, DEFAULT_TAU_DELTA)
                else:
                    chosen_tau, tau_calibrated = 0.0, False
            else:
                chosen_tau, tau_calibrated = 0.0, False

            record = {
                "target_token": target_token,
                "target_prompt": target_prompt,
                "contrast_prompts": contrast_prompts,
                "layers": applied_layers,
                "feats": applied_feats,
                "alpha": alpha,
                "tau": chosen_tau,
                "tau_calibrated": tau_calibrated,
                "elapsed_s": time.time() - t0,
            }
            self.edits.append(record)
            return {"record": record, "diagnostics": diagnostics, "tau_trail": tau_trail}

    def infer(self, prompt: str, top_k: int = 10):
        with self.lock:
            t0 = time.time()
            preds = self.v.infer(prompt, top_k_predictions=top_k)
            return {
                "predictions": [{"token": t, "prob": float(p)} for t, p in preds],
                "elapsed_s": time.time() - t0,
            }

    def reset(self):
        with self.lock:
            zero = [0.0] * self.v.hidden_size
            cleared = 0
            for ed in self.edits:
                for L, feat in zip(ed["layers"], ed["feats"]):
                    self.v.set_gate_vector(L, feat, zero)
                    self.v.set_up_vector(L, feat, zero)
                    self.v.set_down_vector(L, feat, zero)
                    self.v.set_gate_threshold(L, feat, 0.0)
                    self.v.set_feature_meta(L, feat, "", 0.0)
                    cleared += 1
            self.edits = []
            return {"cleared_slots": cleared}

    def status(self):
        return {"active_edits": self.edits}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[daemon] {self.address_string()} - {fmt % args}")

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        n = int(self.headers.get("Content-Length", "0"))
        if not n:
            return {}
        return json.loads(self.rfile.read(n))

    def do_GET(self):
        try:
            if self.path == "/status":
                self._send_json(200, self.server.state.status())
            elif self.path == "/health":
                self._send_json(200, {"ok": True})
            else:
                self._send_json(404, {"error": "unknown path"})
        except Exception as e:
            self._send_json(500, {"error": str(e), "trace": traceback.format_exc()})

    def do_POST(self):
        try:
            body = self._read_json()
            if self.path == "/infer":
                out = self.server.state.infer(body["prompt"], int(body.get("top_k", 10)))
                self._send_json(200, out)
            elif self.path == "/edit":
                out = self.server.state.edit(
                    body["target_prompt"],
                    body["target_token"],
                    body.get("contrast_prompts", []),
                    body.get("layers"),
                    float(body.get("alpha", DEFAULT_ALPHA)),
                    tau=body.get("tau"),
                    calibrate=bool(body.get("calibrate", True)),
                )
                self._send_json(200, out)
            elif self.path == "/reset":
                self._send_json(200, self.server.state.reset())
            else:
                self._send_json(404, {"error": "unknown path"})
        except Exception as e:
            self._send_json(500, {"error": str(e), "trace": traceback.format_exc()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--vindex", default=DEFAULT_VINDEX)
    args = ap.parse_args()

    state = LarqlState(args.vindex)
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    server.state = state
    print(f"[daemon] listening on http://127.0.0.1:{args.port}")
    print(f"[daemon] endpoints: POST /infer  POST /edit  POST /reset  GET /status  GET /health")
    server.serve_forever()


if __name__ == "__main__":
    main()
