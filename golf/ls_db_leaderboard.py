#!/usr/bin/env python3
"""Per-annotator leaderboard read STRAIGHT FROM the Label Studio SQLite DB — no export, no API.

The export API re-serializes every annotated task on each call (minutes on a 20k-task project). The
DB already has the answer: one indexed GROUP BY over the annotations table (`task_completion`),
milliseconds. This opens the DB READ-ONLY (safe while LS runs), auto-detects the schema, and prints
the same leaderboard as ls_progress.py. The functions here also power ls_leaderboard_server.py --db.

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


def find_db(explicit=""):
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
    for root in [Path.home() / ".local/share/label-studio", Path.home()]:
        if root.is_dir():
            hit = next(iter(sorted(root.glob("**/label_studio.sqlite3"))), None)
            if hit:
                return hit
    return None


def connect_ro(db):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)  # read-only: safe while LS is writing
    con.execute("PRAGMA busy_timeout=5000")
    return con


def _cols(con, table):
    return [r[1] for r in con.execute(f"PRAGMA table_info('{table}')")]


def _tables(con):
    return [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]


def _pick(con, prefer, must_have):
    tbs = set(_tables(con))
    for name in prefer:
        if name in tbs and all(c in _cols(con, name) for c in must_have):
            return name
    for name in tbs:
        if all(c in _cols(con, name) for c in must_have):
            return name
    return None


def detect_schema(con):
    """Locate the annotation / user / project / task tables and build the project + name SQL exprs."""
    ann = _pick(con, ["task_completion", "annotation", "tasks_annotation"], ["completed_by_id", "was_cancelled"])
    usr = _pick(con, ["htx_user", "users_user", "auth_user"], ["email"])
    proj = _pick(con, ["project"], ["id", "title"])
    task = _pick(con, ["task"], ["id", "project_id"])
    if not ann or not usr:
        raise RuntimeError(f"schema not recognized (annotations={ann}, users={usr})")
    if "project_id" in _cols(con, ann):
        proj_expr, join = "a.project_id", ""
    elif task:
        proj_expr, join = "t.project_id", f" JOIN {task} t ON t.id = a.task_id"
    else:
        raise RuntimeError("annotations have no project_id and no task table found")
    name_expr = ("TRIM(COALESCE(u.first_name,'')||' '||COALESCE(u.last_name,''))"
                 if "first_name" in _cols(con, usr) else "u.email")
    ann_cols = _cols(con, ann)
    time_col = "updated_at" if "updated_at" in ann_cols else ("created_at" if "created_at" in ann_cols else None)
    return {"ann": ann, "usr": usr, "proj": proj, "task": task, "time_col": time_col,
            "proj_expr": proj_expr, "join": join, "name_expr": name_expr}


def active_annotators(con, schema, project_ids=None, window_secs=60):
    """Distinct annotators who touched an annotation in the last `window_secs` (None if no timestamp col).
    Uses the annotation's timestamp vs SQLite datetime('now') — both are UTC, so they line up."""
    if not schema["time_col"]:
        return None
    where = f"a.was_cancelled=0 AND a.{schema['time_col']} >= datetime('now', ?)"
    params = [f"-{int(window_secs)} seconds"]
    if project_ids:
        where += f" AND {schema['proj_expr']} IN ({','.join('?' * len(project_ids))})"
        params += list(project_ids)
    q = f"SELECT COUNT(DISTINCT a.completed_by_id) FROM {schema['ann']} a{schema['join']} WHERE {where}"
    return con.execute(q, params).fetchone()[0]


def user_counts(con, schema, project_ids=None):
    """-> list of (pid, uid, name, n) for non-cancelled annotations, most first."""
    where, params = "a.was_cancelled = 0", []
    if project_ids:
        where += f" AND {schema['proj_expr']} IN ({','.join('?' * len(project_ids))})"
        params = list(project_ids)
    ne = schema["name_expr"]
    q = (f"SELECT {schema['proj_expr']} pid, a.completed_by_id uid, "
         f"CASE WHEN {ne}='' OR {ne} IS NULL THEN u.email ELSE {ne} END nm, COUNT(*) n "
         f"FROM {schema['ann']} a{schema['join']} LEFT JOIN {schema['usr']} u ON u.id = a.completed_by_id "
         f"WHERE {where} GROUP BY pid, uid ORDER BY n DESC")
    return con.execute(q, params).fetchall()


def project_progress(con, schema, project_ids=None):
    """-> {pid: {'title','total','done'}}. total=all tasks, done=tasks with >=1 non-cancelled ann."""
    task = schema["task"]
    titles = {r[0]: r[1] for r in con.execute(f"SELECT id, title FROM {schema['proj']}")} if schema["proj"] else {}
    out = {}
    ids = project_ids or list(titles)
    for pid in ids:
        total = con.execute(f"SELECT COUNT(*) FROM {task} WHERE project_id=?", (pid,)).fetchone()[0] if task else 0
        done = con.execute(
            f"SELECT COUNT(DISTINCT a.task_id) FROM {schema['ann']} a{schema['join']} "
            f"WHERE a.was_cancelled=0 AND {schema['proj_expr']}=?", (pid,)).fetchone()[0]
        out[pid] = {"title": titles.get(pid, f"project {pid}"), "total": total, "done": done}
    return out


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
    con = connect_ro(db)
    print(f"db: {db}")
    try:
        schema = detect_schema(con)
    except RuntimeError as e:
        if args.debug:
            for t in _tables(con):
                print(f"  {t}: {_cols(con, t)}")
        sys.exit(f"{e}. Re-run with --debug and send the output.")
    if args.debug:
        print("  schema:", {k: schema[k] for k in ("ann", "usr", "proj", "task")})

    rows = user_counts(con, schema, args.project)
    by_proj, by_user, total = {}, {}, 0
    for pid, uid, nm, n in rows:
        nm = nm or f"user{uid}"
        by_proj.setdefault(pid, []).append((nm, n))
        by_user[nm] = by_user.get(nm, 0) + n
        total += n
    for pid in sorted(by_proj):
        prog = project_progress(con, schema, [pid])[pid]
        print(f"\n=== project {pid}: {prog['title']} — {prog['done']:,}/{prog['total']:,} tasks ===")
        for nm, n in sorted(by_proj[pid], key=lambda x: -x[1]):
            print(f"  {nm:24s} {n:>7,}")
    active = active_annotators(con, schema, args.project, 300)
    print(f"\n{'='*46}\nLEADERBOARD (all requested projects) — {total:,} annotations"
          + (f"  |  🟢 {active} labeling now" if active is not None else ""))
    for rank, (nm, n) in enumerate(sorted(by_user.items(), key=lambda x: -x[1]), 1):
        print(f"  {rank:>2}. {nm:24s} {n:>7,}")


if __name__ == "__main__":
    main()
