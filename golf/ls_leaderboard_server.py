#!/usr/bin/env python3
"""Live Label Studio leaderboard as a web page the whole team can open.

Runs a tiny web server on this box that polls the LS API in the background and serves an auto-
refreshing HTML leaderboard (per-annotator counts + overall progress). Anyone on the network opens
http://<this-box-ip>:<port>/ — no token needed on their side (the server holds it).

    # DB mode (FAST — reads the LS SQLite directly, per-person is instant; run ON the LS box):
    python golf/ls_leaderboard_server.py --db --project 15 18 20 --port 8090
    # API mode (when not on the LS box): needs a token, per-person via the slow export
    python golf/ls_leaderboard_server.py --url http://105.145.25.32:8080 --token <TOKEN> \
        --project 15 18 20 --port 8090 --refresh 120
    # team opens:  http://105.145.25.32:8090/
"""
import argparse
import html
import os
import sys
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # import the sibling DB helpers
from ls_db_leaderboard import (active_annotators, connect_ro, detect_schema, find_db,
                               project_progress, user_counts)

STATE = {"html": "<h1>starting…</h1>", "ts": 0}
ARGS = None
DB_MODE = False
LOCK = threading.Lock()


def ensure_fresh():
    """DB mode: recompute on demand (so a manual F5 is truly live), with a small TTL so a burst of
    viewers coalesces into one DB read."""
    ttl = max(2, min(ARGS.refresh, 5))
    if STATE["ts"] and time.time() - STATE["ts"] < ttl:
        return
    with LOCK:
        if STATE["ts"] and time.time() - STATE["ts"] < ttl:
            return
        try:
            STATE["html"] = render(*db_stats())
            STATE["ts"] = time.time()
        except Exception as e:
            if not STATE["ts"]:
                STATE["html"] = ("<h1>⛳ Golf Labeling Summary</h1>"
                                 f"<p>error reading the LS DB: {html.escape(str(e))}</p>")


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


def fast_stats():
    """Cheap: users + per-project done/total from the project summary (no export). Renders instantly."""
    users = {}
    for u in get("/api/users/"):
        nm = (f"{u.get('first_name','')} {u.get('last_name','')}").strip() or u.get("email") or f"user{u['id']}"
        users[u["id"]] = nm
    pl = get("/api/projects/")
    available = {p["id"] for p in (pl.get("results", pl) if isinstance(pl, dict) else pl)}
    projects = []
    g_done = g_total = 0
    for pid in ARGS.project:
        if pid not in available:
            projects.append((pid, f"project {pid} — NOT FOUND (ids: {sorted(available)})", 0, 0))
            continue
        proj = get(f"/api/projects/{pid}/")
        total, done = proj.get("task_number", 0), proj.get("num_tasks_with_annotations", 0)
        g_total += total; g_done += done
        projects.append((pid, proj.get("title", f"project {pid}"), done, total))
    return users, projects, g_done, g_total


def count_project(pid):
    """Per-user annotation counts for ONE project. Uses the LIGHT JSON_MIN export (annotator id only,
    no box geometry) — much smaller/faster than full JSON on big projects (train = 23k+ annotated)."""
    c = Counter()
    for r in (get(f"/api/projects/{pid}/export", exportType="JSON_MIN") or []):
        uid = r.get("annotator")
        if uid is None:  # fallback if a build emits full-shape rows
            for a in (r.get("annotations") or []):
                if a.get("was_cancelled"):
                    continue
                u = a.get("completed_by")
                c[u.get("id") if isinstance(u, dict) else u] += 1
        else:
            c[uid] += 1
    return c


def db_stats():
    """Everything (overall + per-person) straight from the LS SQLite — instant. No token, no export."""
    db = find_db(ARGS.db if ARGS.db is not True else "")
    if not db or not db.is_file():
        raise RuntimeError("label_studio.sqlite3 not found — pass --db <path>")
    con = connect_ro(db)
    try:
        schema = detect_schema(con)
        prog = project_progress(con, schema, ARGS.project)
        ids = ARGS.project or sorted(prog)
        rows = user_counts(con, schema, ARGS.project)
        active = active_annotators(con, schema, ARGS.project, ARGS.active_window)
    finally:
        con.close()
    users, by_user = {}, Counter()
    for pid, uid, nm, n in rows:
        users[uid] = nm or f"user{uid}"
        by_user[uid] += n
    projects = [(pid, prog[pid]["title"], prog[pid]["done"], prog[pid]["total"]) for pid in ids]
    g_done = sum(prog[pid]["done"] for pid in ids)
    g_total = sum(prog[pid]["total"] for pid in ids)
    return users, by_user, projects, g_done, g_total, active


def _winlbl(secs):
    return f"{secs // 60} min" if secs >= 120 else f"{secs}s"


def render(users, by_user, projects, g_done, g_total, active=None, computing=False):
    pct = 100 * g_done / g_total if g_total else 0
    rows = ""
    top = by_user.most_common()
    mx = top[0][1] if top else 1
    for rank, (uid, c) in enumerate(top, 1):
        name = html.escape(users.get(uid, f"user{uid}"))
        w = 100 * c / mx if mx else 0
        rows += (f'<tr><td class="rk">{rank}</td><td class="nm">{name}</td>'
                 f'<td class="ct">{c:,}</td><td class="bar"><span style="width:{w:.1f}%"></span></td></tr>')
    if not rows:
        rows = (f'<tr><td colspan="4" style="text-align:center;color:var(--soft)">'
                f'{"computing per-person counts…" if computing else "no annotations yet"}</td></tr>')
    proj_rows = "".join(
        f'<div class="pj"><b>{html.escape(t)}</b> — {d:,}/{n:,} '
        f'<small>({100*d/n if n else 0:.1f}%)</small></div>' for _, t, d, n in projects)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="{ARGS.refresh}">
<title>Golf Labeling Summary</title>
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
 tr:last-child td{{border-bottom:0}}
 .rk{{width:44px;text-align:center;font-size:15px;font-weight:600;color:var(--soft)}}
 .nm{{font-weight:600}} .ct{{width:90px;text-align:right;font-weight:800;color:var(--g2);font-size:17px}}
 .bar{{width:38%}} .bar>span{{display:block;height:9px;border-radius:99px;
  background:linear-gradient(90deg,var(--g),var(--g2))}}
 .foot{{color:var(--soft);font-size:12px;margin-top:14px;text-align:center}}
 .live{{display:inline-flex;align-items:center;gap:6px;background:#e9f6ef;color:var(--g2);
  font-weight:700;padding:3px 10px;border-radius:99px;font-size:13px}}
 .dot{{width:8px;height:8px;border-radius:50%;background:#22c55e;box-shadow:0 0 0 0 rgba(34,197,94,.5);
  animation:pulse 1.6s infinite}} @keyframes pulse{{70%{{box-shadow:0 0 0 7px rgba(34,197,94,0)}}}}
</style></head><body><div class="wrap">
 <h1>⛳ Golf Labeling Summary</h1>
 <p class="sub">Live — auto-refreshes every {ARGS.refresh}s</p>
 <div class="prog"><div class="big">{g_done:,} <span style="font-size:16px;color:var(--soft);font-weight:500">/ {g_total:,} labeled ({pct:.1f}%)</span>
  {'' if active is None else f'<span class="live"><span class="dot"></span>{active} labeling now <span style="font-weight:500;opacity:.75">· last {_winlbl(ARGS.active_window)}</span></span>'}</div>
  <div class="track"><span></span></div>
  <div class="pjs">{proj_rows}</div></div>
 <table>{rows}</table>
 <p class="foot">updated {time.strftime('%H:%M:%S')} · counts human annotations only</p>
</div></body></html>"""


def refresher():
    by_user = Counter()       # last computed per-person counts (kept between cheap refreshes)
    people_ts = 0.0           # when we last did the heavy per-person export
    while True:
        try:
            # 1) cheap pass EVERY cycle: overall progress + project bars (fast project-summary calls)
            users, projects, g_done, g_total = fast_stats()
            first = not by_user and people_ts == 0
            STATE["html"] = render(users, by_user, projects, g_done, g_total, computing=first)
            STATE["ts"] = time.time()
            # 2) heavy pass, only every --people-refresh secs: re-export per-person counts
            if time.time() - people_ts >= ARGS.people_refresh:
                fresh = Counter()
                for pid, title, done, total in projects:
                    if "NOT FOUND" in title:
                        continue
                    fresh.update(count_project(pid))
                    STATE["html"] = render(users, fresh, projects, g_done, g_total)  # fill as each finishes
                    STATE["ts"] = time.time()
                by_user = fresh
                people_ts = time.time()
        except Exception as e:
            # keep the last good page if we have one; only show a bare error before the first success
            if STATE["ts"] == 0:
                STATE["html"] = ("<h1>⛳ Golf Labeling Summary</h1>"
                                 f"<p>error talking to Label Studio: {html.escape(str(e))}</p>")
        time.sleep(max(15, ARGS.refresh))


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if DB_MODE:
            ensure_fresh()               # F5 = live (TTL-coalesced); API mode serves the bg snapshot
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
    ap.add_argument("--db", nargs="?", const=True, default=None,
                    help="DB MODE (fast): read the LS SQLite directly. Bare --db auto-finds it, or give a "
                         "path. Run on the LS box. No token needed. Recommended.")
    ap.add_argument("--url", help="API MODE: LS base URL (use when NOT on the LS box)")
    ap.add_argument("--token", help="API MODE: LS API token")
    ap.add_argument("--project", type=int, nargs="*", help="project id(s); in DB mode, omit for all")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--refresh", type=int, default=60, help="page/poll seconds")
    ap.add_argument("--active-window", type=int, default=300,
                    help="DB mode: seconds counted as 'labeling now' (distinct annotators active in this "
                         "window). 300s=5min captures people mid-review who haven't submitted yet")
    ap.add_argument("--people-refresh", type=int, default=600,
                    help="API mode only: per-person recount seconds (heavy export; keep >> --refresh)")
    ARGS = ap.parse_args()

    global DB_MODE
    DB_MODE = ARGS.db is not None or not ARGS.url
    if DB_MODE:
        if not ARGS.db:                      # `--url` omitted and no `--db` -> default to auto-find DB
            ARGS.db = True
        print("mode: DB (reading LS SQLite directly, live on each request)", flush=True)
        ensure_fresh()                       # warm the first page
    else:
        if not ARGS.token or not ARGS.project:
            ap.error("API mode needs --token and --project")
        print("mode: API (via export)", flush=True)
        threading.Thread(target=refresher, daemon=True).start()
        time.sleep(1)
    srv = ThreadingHTTPServer(("0.0.0.0", ARGS.port), H)
    print(f"leaderboard on http://0.0.0.0:{ARGS.port}/  (team opens http://<this-box-ip>:{ARGS.port}/)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
