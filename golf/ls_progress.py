#!/usr/bin/env python3
"""Label Studio leaderboard — how many tasks each annotator has labeled, per project + overall.

Counts human ANNOTATIONS (not the imported model predictions) by completed_by, so you see exactly
who did what and overall progress. Works on the OSS edition.

    python golf/ls_progress.py --url http://105.145.25.32:8080 --token <API_TOKEN> --project 1
    python golf/ls_progress.py --url ... --token ... --project 1 2 3      # combine several projects

Get the token in LS: click your avatar -> Account & Settings -> Access Token. Project id = the number
in the project URL (/projects/<id>/).
"""
import argparse
from collections import Counter

import requests


# Three LS token flavors: legacy "Access Token" (`Token <t>`), JWT access token (`Bearer <t>`),
# and JWT REFRESH token (what the UI shows in JWT mode, token_type=refresh) -> exchange via
# POST /api/token/refresh for a short-lived access token, then Bearer it. Auto-detect + re-mint on 401.
AUTH = {"scheme": None, "bearer": None}


def _refresh_access(url, token):
    r = requests.post(url.rstrip("/") + "/api/token/refresh", json={"refresh": token}, timeout=30)
    r.raise_for_status()
    AUTH["bearer"] = r.json()["access"]


def get(url, token, path, **params):
    def _try(hdr):
        return requests.get(url.rstrip("/") + path, headers=hdr, params=params, timeout=60)

    if AUTH["scheme"] == "Token":
        r = _try({"Authorization": f"Token {token}"})
    elif AUTH["scheme"] == "Bearer":
        r = _try({"Authorization": f"Bearer {AUTH['bearer']}"})
        if r.status_code == 401:
            _refresh_access(url, token)
            r = _try({"Authorization": f"Bearer {AUTH['bearer']}"})
    else:
        r = _try({"Authorization": f"Token {token}"})
        if r.status_code != 401:
            AUTH["scheme"] = "Token"
        else:
            r = _try({"Authorization": f"Bearer {token}"})
            if r.status_code != 401:
                AUTH["scheme"], AUTH["bearer"] = "Bearer", token
            else:
                _refresh_access(url, token)
                r = _try({"Authorization": f"Bearer {AUTH['bearer']}"})
                AUTH["scheme"] = "Bearer"
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="Label Studio base URL, e.g. http://105.145.25.32:8080")
    ap.add_argument("--token", required=True, help="your LS API token (Account & Settings -> Access Token)")
    ap.add_argument("--project", type=int, nargs="+", help="project id(s); omit to just list projects")
    ap.add_argument("--page-size", type=int, default=1000)
    args = ap.parse_args()

    # what projects exist? (validates auth + shows the real ids)
    pl = get(args.url, args.token, "/api/projects/")
    plist = pl.get("results", pl) if isinstance(pl, dict) else pl
    available = {p["id"]: p.get("title", "") for p in plist}
    if not args.project:
        print("projects on this server (use the id with --project):")
        for pid, title in available.items():
            print(f"  {pid:>4}  {title}")
        return
    for pid in args.project:
        if pid not in available:
            print(f"⚠ project {pid} not found. available: "
                  + ", ".join(f"{i}={t}" for i, t in available.items()))
            return

    # user id -> name
    users = {}
    try:
        for u in get(args.url, args.token, "/api/users/"):
            nm = (f"{u.get('first_name','')} {u.get('last_name','')}").strip() or u.get("email") or f"user{u['id']}"
            users[u["id"]] = nm
    except Exception as e:
        print(f"(couldn't list users: {e}; showing user ids)")

    by_user = Counter()
    grand_done = grand_total = 0
    for pid in args.project:
        proj = get(args.url, args.token, f"/api/projects/{pid}/")
        total = proj.get("task_number", 0)
        done = proj.get("num_tasks_with_annotations", 0)
        grand_total += total; grand_done += done
        # paginate tasks, count annotations by completed_by
        page = 1
        pcount = Counter()
        while True:
            data = get(args.url, args.token, "/api/tasks/", project=pid, page=page, page_size=args.page_size)
            tasks = data.get("tasks", data) if isinstance(data, dict) else data
            if not tasks:
                break
            for t in tasks:
                for a in (t.get("annotations") or []):
                    if a.get("was_cancelled"):
                        continue
                    uid = a.get("completed_by")
                    uid = uid.get("id") if isinstance(uid, dict) else uid
                    pcount[uid] += 1
            if len(tasks) < args.page_size:
                break
            page += 1
        by_user.update(pcount)
        pct = 100 * done / total if total else 0
        print(f"\n=== project {pid}: {proj.get('title','')} — {done:,}/{total:,} tasks done ({pct:.1f}%) ===")
        for uid, c in pcount.most_common():
            print(f"  {users.get(uid, f'user{uid}'):24s} {c:>7,}")

    print(f"\n{'='*46}\nLEADERBOARD (all projects) — {grand_done:,}/{grand_total:,} done "
          f"({100*grand_done/grand_total if grand_total else 0:.1f}%)")
    for rank, (uid, c) in enumerate(by_user.most_common(), 1):
        print(f"  {rank:>2}. {users.get(uid, f'user{uid}'):24s} {c:>7,}")


if __name__ == "__main__":
    main()
