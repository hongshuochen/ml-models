#!/usr/bin/env python3
"""Per-annotator leaderboard read STRAIGHT FROM the Label Studio SQLite DB — no export, no API.

The export API re-serializes every annotated task on each call (minutes on a 20k-task project). The
DB already has the answer: one indexed GROUP BY over the annotations table (`task_completion`),
milliseconds. This opens the DB READ-ONLY (safe while LS runs), auto-detects the schema, and prints
the same leaderboard as ls_progress.py.

    ~/ml-models/.venv/bin/python golf/ls_db_leaderboard.py            # auto-find DB, all projects
    ~/ml-models/.venv/bin/python golf/ls_db_leaderboard.py --project 15 18 20
    ~/ml-models/.venv/bin/python golf/ls_db_leaderboard.py --db /path/to/label_studio.sqlite3

Default SQLite lives at $LABEL_STUDIO_BASE_DATA_DIR/label_studio.sqlite3 or
~/.local/share/label-studio/label_studio.sqlite3.
"""
import argparse
import os
import sqlite3
import sys
from pathlib import Path


def find_db(explicit):
    if explicit:
        return Path(explicit).expanduser()
    cands = []
    env = os.environ.get("LABEL_STUDIO_BASE_DATA_DIR")
    if env:
        cands.append(Path(env) / "label_studio.sqlite3")
    cands += [
        Path.home() / ".local/share/label-studio/label_studio.sqlite3",
        Path.home() / "label_studio.sqlite3",
        Path.cwd() / "label_studio.sqlite3",
    ]
    for c in cands:
        if c.is_file():
            return c
    # last resort: shallow search of common roots
    for root in [Path.home() / ".local/share/label-studio", Path.home()]:
        if root.is_dir():
            hit = next(iter(sorted(root.glob("**/label_studio.sqlite3"))), None)
            if hit:
                return hit
    return None


def cols(con, table):
    return [r[1] for r in con.execute(f"PRAGMA table_info('{table}')")]


def tables(con):
    return [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]


def pick(con, want_any_of, must_have):
    """First table whose columns include all of must_have (and exists in the candidate name list)."""
    tbs = set(tables(con))
    for name in want_any_of:
        if name in tbs and all(c in cols(con, name) for c in must_have):
            return name
    for name in tbs:  # fall back to scanning every table for the required columns
        if all(c in cols(con, name) for c in must_have):
            return name
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="", help="path to label_studio.sqlite3 (auto-detected if omitted)")
    ap.add_argument("--project", type=int, nargs="*", help="project id(s); omit for all")
    ap.add_argument("--debug", action="store_true", help="print detected tables/columns")
    args = ap.parse_args()

    db = find_db(args.db)
    if not db or not db.is_file():
        sys.exit("could not find label_studio.sqlite3 — pass --db /path/to/it "
                 "(check the terminal where you ran `label-studio start`, or $LABEL_STUDIO_BASE_DATA_DIR)")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.execute("PRAGMA busy_timeout=5000")
    print(f"db: {db}")

    ann = pick(con, ["task_completion", "annotation", "tasks_annotation"],
               ["completed_by_id", "was_cancelled"])
    usr = pick(con, ["htx_user", "users_user", "auth_user"], ["email"])
    proj = pick(con, ["project"], ["id", "title"])
    if args.debug:
        for t in (ann, usr, proj):
            print(f"  table {t}: {cols(con, t) if t else '—'}")
    if not ann or not usr:
        sys.exit(f"schema not recognized (annotations={ann}, users={usr}). Re-run with --debug and send output.")

    ann_cols = cols(con, ann)
    # project id: annotation table may carry project_id directly, else join through the task table
    if "project_id" in ann_cols:
        proj_expr, join = "a.project_id", ""
    else:
        task = pick(con, ["task"], ["id", "project_id"])
        if not task:
            sys.exit("annotations have no project_id and no task table found; re-run with --debug.")
        proj_expr, join = "t.project_id", f" JOIN {task} t ON t.id = a.task_id"

    where = "a.was_cancelled = 0"
    params = []
    if args.project:
        where += f" AND {proj_expr} IN ({','.join('?' * len(args.project))})"
        params = args.project

    name_expr = ("TRIM(COALESCE(u.first_name,'')||' '||COALESCE(u.last_name,''))"
                 if "first_name" in cols(con, usr) else "u.email")
    q = (f"SELECT {proj_expr} pid, a.completed_by_id uid, "
         f"CASE WHEN {name_expr}='' THEN u.email ELSE {name_expr} END nm, COUNT(*) n "
         f"FROM {ann} a{join} LEFT JOIN {usr} u ON u.id = a.completed_by_id "
         f"WHERE {where} GROUP BY pid, uid ORDER BY n DESC")
    rows = con.execute(q, params).fetchall()

    titles = {r[0]: r[1] for r in con.execute(f"SELECT id, title FROM {proj}")} if proj else {}
    by_proj, by_user, total = {}, {}, 0
    for pid, uid, nm, n in rows:
        by_proj.setdefault(pid, []).append((nm or f"user{uid}", n))
        by_user[nm or f"user{uid}"] = by_user.get(nm or f"user{uid}", 0) + n
        total += n

    for pid in sorted(by_proj):
        print(f"\n=== project {pid}: {titles.get(pid,'')} ===")
        for nm, n in sorted(by_proj[pid], key=lambda x: -x[1]):
            print(f"  {nm:24s} {n:>7,}")
    print(f"\n{'='*46}\nLEADERBOARD (all requested projects) — {total:,} annotations")
    for rank, (nm, n) in enumerate(sorted(by_user.items(), key=lambda x: -x[1]), 1):
        print(f"  {rank:>2}. {nm:24s} {n:>7,}")


if __name__ == "__main__":
    main()
