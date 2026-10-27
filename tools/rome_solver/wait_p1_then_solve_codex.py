#!/usr/bin/env python3
"""Wait for shared C and the ONLY permitted peer file, then approved solves.

No directory scans, peer factors, traces, status.json or source code reads.
An A2 failure stops this waiting chain rather than leaving a latent launch.
"""
import argparse
import subprocess
import time

from p1_runtime import ROOT,REPO,read,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--approved-brief-40',action='store_true');a=p.parse_args()
    if not a.approved_brief_40:raise RuntimeError('Approved scope required')
    deadline=time.monotonic()+8*3600
    while time.monotonic()<deadline:
        for i in range(6):
            status=ROOT/f'covariance/parts-a2/service{i}/status.json'
            if status.exists()and read(status)['status']=='failed':raise RuntimeError('A2 failed; no future solve launch')
        cov=ROOT/'covariance/status.json';peer=ROOT/'solves-claude/completion.json'
        ready=cov.exists()and read(cov).get('status')=='complete'
        complete=ready and peer.exists()and read(peer).get('complete')is True
        write(ROOT/'codex-next-stage.json',dict(covariance_complete=ready,peer_complete=bool(complete),only_peer_file_read=str(peer),own_solves_started=False))
        if complete:
            subprocess.run(['/mnt/models/finetune/venv/bin/python','-u',str(REPO/'tools/rome_solver/launch_p1_solves_codex.py'),'--approved-brief-40'],cwd=REPO,check=True)
            return
        time.sleep(10)
    raise TimeoutError('P1 dependencies not complete after eight hours; no model launched')


if __name__=='__main__':main()
