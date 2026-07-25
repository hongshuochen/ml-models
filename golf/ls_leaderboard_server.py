#!/usr/bin/env python3
"""Live Label Studio leaderboard as a web page the whole team can open.

Runs a tiny web server on this box that polls the LS API in the background and serves an auto-
refreshing HTML leaderboard (per-annotator counts + overall progress). Anyone on the network opens
http://<this-box-ip>:<port>/ — no token needed on their side (the server holds it).

    python golf/ls_leaderboard_server.py --url http://105.145.25.32:8080 --token <TOKEN> \
        --project 1 2 3 --port 8090 --refresh 120
    # team opens:  http://105.145.25.32:8090/
"""
import argparse
import html
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

STATE = {"html": "<h1>starting…</h1>", "ts": 0}
ARGS = None


# Three LS token flavors:
#  - legacy "Access Token"           -> header `Token <t>`
#  - JWT ACCESS token (short-lived)  -> header `Bearer <t>`
#  - JWT REFRESH token (what the UI's "Access Token" page shows in JWT mode, token_type=refresh)
#    -> POST /api/token/refresh {"refresh": t} to mint a short-lived access token, then Bearer it.
# We auto-detect and, for the refresh flow, re-mint on 401 (access tokens expire).
AUTH = {"scheme": None, "bearer": None}  # scheme: "Token" | "Bearer"; bearer: the value to send


def _refresh_access():
    r = requests.post(ARGS.url.rstrip("/") + "/api/token/refresh",
                      json={"refresh": ARGS.token}, timeout=30)
    r.raise_for_status()
    AUTH["bearer"] = r.json()["access"]


def _try(path, params, header):
    return requests.get(ARGS.url.rstrip("/") + path, headers=header, params=params, timeout=300)


def get(path, **params):
    # Established scheme: use it (re-minting the JWT access token on expiry).
    if AUTH["scheme"] == "Token":
        r = _try(path, params, {"Authorization": f"Token {ARGS.token}"})
    elif AUTH["scheme"] == "Bearer":
        r = _try(path, params, {"Authorization": f"Bearer {AUTH['bearer']}"})
        if r.status_code == 401:            # access token expired -> re-mint and retry
            _refresh_access()
            r = _try(path, params, {"Authorization": f"Bearer {AUTH['bearer']}"})
    else:
        # First call: probe legacy Token, then raw Bearer, then refresh->access Bearer.
        r = _try(path, params, {"Authorization": f"Token {ARGS.token}"})
        if r.status_code != 401:
            AUTH["scheme"] = "Token"
        else:
            r = _try(path, params, {"Authorization": f"Bearer {ARGS.token}"})
            if r.status_code != 401:
                AUTH["scheme"], AUTH["bearer"] = "Bearer", ARGS.token
            else:
                _refresh_access()           # treat token as a refresh token
                r = _try(path, params, {"Authorization": f"Bearer {AUTH['bearer']}"})
                AUTH["scheme"] = "Bearer"
    r.raise_for_status()
    return r.json()


def build_stats():
    users = {}
    for u in get("/api/users/"):
        nm = (f"{u.get('first_name','')} {u.get('last_name','')}").strip() or u.get("email") or f"user{u['id']}"
        users[u["id"]] = nm
    pl = get("/api/projects/")
    available = {p["id"] for p in (pl.get("results", pl) if isinstance(pl, dict) else pl)}
    by_user, projects = Counter(), []
    g_done = g_total = 0
    for pid in ARGS.project:
        if pid not in available:
            projects.append((pid, f"project {pid} — NOT FOUND (ids: {sorted(available)})", 0, 0))
            continue
        proj = get(f"/api/projects/{pid}/")
        total, done = proj.get("task_number", 0), proj.get("num_tasks_with_annotations", 0)
        g_total += total; g_done += done
        projects.append((pid, proj.get("title", f"project {pid}"), done, total))
        # /api/tasks/ list doesn't embed annotations -> export gives only annotated tasks with full anns
        for t in (get(f"/api/projects/{pid}/export", exportType="JSON") or []):
            for a in (t.get("annotations") or []):
                if a.get("was_cancelled"):
                    continue
                uid = a.get("completed_by")
                uid = uid.get("id") if isinstance(uid, dict) else uid
                by_user[uid] += 1
    return users, by_user, projects, g_done, g_total


MEDAL = {1: "\U0001F947", 2: "\U0001F948", 3: "\U0001F949"}


def render(users, by_user, projects, g_done, g_total):
    pct = 100 * g_done / g_total if g_total else 0
    rows = ""
    top = by_user.most_common()
    mx = top[0][1] if top else 1
    for rank, (uid, c) in enumerate(top, 1):
        name = html.escape(users.get(uid, f"user{uid}"))
        w = 100 * c / mx if mx else 0
        medal = MEDAL.get(rank, f"{rank}")
        cls = " top" if rank <= 3 else ""
        rows += (f'<tr class="r{cls}"><td class="rk">{medal}</td><td class="nm">{name}</td>'
                 f'<td class="ct">{c:,}</td><td class="bar"><span style="width:{w:.1f}%"></span></td></tr>')
    proj_rows = "".join(
        f'<div class="pj"><b>{html.escape(t)}</b> — {d:,}/{n:,} '
        f'<small>({100*d/n if n else 0:.1f}%)</small></div>' for _, t, d, n in projects)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="{ARGS.refresh}">
<title>Golf Labeling Leaderboard</title>
<style>
 :root{{--g:#14603f;--g2:#1c7a4f;--ink:#12201a;--soft:#5a6b61;--bg:#f3f8f5;--card:#fff;--line:#dbe4de}}
 *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,Segoe UI,Roboto,sans-serif;padding:clamp(16px,4vw,40px)}}
 .wrap{{max-width:760px;margin:0 auto}}
 h1{{font-size:26px;margin:0 0 2px}} .sub{{color:var(--soft);font-size:13px;margin:0 0 18px}}
 .prog{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin-bottom:18px}}
 .big{{font-size:34px;font-weight:800;color:var(--g2);font-variant-numeric:tabular-nums}}
 .track{{height:14px;background:#e6ede9;border-radius:99px;overflow:hidden;margin:10px 0 4px}}
 .track>span{{display:block;height:100%;background:linear-gradient(90deg,var(--g),var(--g2));width:{pct:.1f}%}}
 .pjs{{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:8px;font-size:13px;color:var(--soft)}}
 table{{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);
  border-radius:14px;overflow:hidden}}
 td{{padding:11px 14px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}}
 tr:last-child td{{border-bottom:0}} tr.top{{background:#f0f8f3}}
 .rk{{width:44px;text-align:center;font-size:19px;font-weight:700;color:var(--soft)}}
 .nm{{font-weight:600}} .ct{{width:90px;text-align:right;font-weight:800;color:var(--g2);font-size:17px}}
 .bar{{width:38%}} .bar>span{{display:block;height:9px;border-radius:99px;
  background:linear-gradient(90deg,var(--g),var(--g2))}}
 .foot{{color:var(--soft);font-size:12px;margin-top:14px;text-align:center}}
</style></head><body><div class="wrap">
 <h1>⛳ Golf Labeling Leaderboard</h1>
 <p class="sub">Live — auto-refreshes every {ARGS.refresh}s</p>
 <div class="prog"><div class="big">{g_done:,} <span style="font-size:16px;color:var(--soft);font-weight:500">/ {g_total:,} labeled ({pct:.1f}%)</span></div>
  <div class="track"><span></span></div>
  <div class="pjs">{proj_rows}</div></div>
 <table>{rows or '<tr><td>no annotations yet</td></tr>'}</table>
 <p class="foot">updated {time.strftime('%H:%M:%S')} · counts human annotations only</p>
</div></body></html>"""


def refresher():
    while True:
        try:
            STATE["html"] = render(*build_stats())
            STATE["ts"] = time.time()
        except Exception as e:
            STATE["html"] = f"<h1>Golf Leaderboard</h1><p>error talking to Label Studio: {html.escape(str(e))}</p>"
        time.sleep(max(15, ARGS.refresh))


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        body = STATE["html"].encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--project", type=int, nargs="+", required=True)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--refresh", type=int, default=120, help="poll/refresh seconds")
    ARGS = ap.parse_args()

    threading.Thread(target=refresher, daemon=True).start()
    time.sleep(1)
    srv = ThreadingHTTPServer(("0.0.0.0", ARGS.port), H)
    print(f"leaderboard on http://0.0.0.0:{ARGS.port}/  (team opens http://<this-box-ip>:{ARGS.port}/)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
