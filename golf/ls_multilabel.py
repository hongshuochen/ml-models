#!/usr/bin/env python3
"""How many tasks were annotated MORE THAN ONCE (overlap / double-labeled), per project — from the LS DB.

Groups non-cancelled annotations by task and reports the distribution (1 ann, 2 anns, …) plus the
% of labeled tasks that have ≥2. Read-only, ms.

    ~/ml-models/.venv/bin/python golf/ls_multilabel.py --project 15
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ls_db_leaderboard import connect_ro, detect_schema, find_db


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="")
    ap.add_argument("--project", type=int, nargs="+", required=True)
    args = ap.parse_args()

    db = find_db(args.db)
    if not db or not db.is_file():
        sys.exit("could not find label_studio.sqlite3 — pass --db /path/to/it")
    con = connect_ro(db)
    schema = detect_schema(con)
    print(f"db: {db}")

    for pid in args.project:
        q = (f"SELECT cnt, COUNT(*) FROM ("
             f"  SELECT a.task_id, COUNT(*) cnt FROM {schema['ann']} a{schema['join']} "
             f"  WHERE a.was_cancelled=0 AND {schema['proj_expr']}=? GROUP BY a.task_id"
             f") GROUP BY cnt ORDER BY cnt")
        dist = con.execute(q, (pid,)).fetchall()          # [(anns_per_task, num_tasks), ...]
        labeled = sum(n for _, n in dist)
        multi = sum(n for cnt, n in dist if cnt >= 2)
        print(f"\n=== project {pid} ===")
        print(f"  labeled tasks (>=1 annotation): {labeled:,}")
        for cnt, n in dist:
            print(f"    {cnt} annotation{'s' if cnt > 1 else ' '}: {n:,} tasks")
        pct = 100 * multi / labeled if labeled else 0
        print(f"  >1 annotation: {multi:,} / {labeled:,} labeled = {pct:.2f}%")


if __name__ == "__main__":
    main()
