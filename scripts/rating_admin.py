#!/usr/bin/env python3
"""Admin for the ongiini.ai/rate/ surface. Runs inside the webhook container:

    docker exec -i ongiini-webhook python3 /data/private/rating_admin.py add-rater "Ndapewa" oshikwanyama
    docker exec -i ongiini-webhook python3 /data/private/rating_admin.py load /data/private/rating_items_r1.json
    docker exec -i ongiini-webhook python3 /data/private/rating_admin.py add-reviewer "Ndapewa" oshikwanyama v1_review
    docker exec -i ongiini-webhook python3 /data/private/rating_admin.py review-progress v1_review
    docker exec -i ongiini-webhook python3 /data/private/rating_admin.py open-rating off
    docker exec -i ongiini-webhook python3 /data/private/rating_admin.py report
    docker exec -i ongiini-webhook python3 /data/private/rating_admin.py raters
    docker exec -i ongiini-webhook python3 /data/private/rating_admin.py deactivate 3
    docker exec -i ongiini-webhook python3 /data/private/rating_admin.py delete-rater 3
    docker exec -i ongiini-webhook python3 /data/private/rating_admin.py delete-round demo
    docker exec -i ongiini-webhook python3 /data/private/rating_admin.py reset-test --yes
    docker exec -i ongiini-webhook python3 /data/private/rating_admin.py block 7
    docker exec -i ongiini-webhook python3 /data/private/rating_admin.py block-number 264811234567

Most raters get their link themselves by asking Ongiini AI on WhatsApp
(tools/rate.py). block / block-number keep someone out — e.g. the
reference translator, who must not rate her own translations;
block-number works before the number ever asks.

Review answers are protected: delete-rater, delete-round and reset-test
refuse while they would delete any, unless --delete-review-answers is given.

delete-rater removes a rater and all their ratings; delete-round removes a
round's items and their ratings; reset-test wipes everything (test phase
only — refuses without --yes).

add-reviewer invites a native speaker to check EVERY review item of one
dialect in a round (build items with scripts/build_review_items.py).

open-rating off pauses the open rating (no new WhatsApp links, open raters
get no items) so only reviewers work; open-rating on resumes it.

add-rater prints the personal invite link once — send it via WhatsApp.
Only a hash of the token is stored; a lost link is replaced by a new rater.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if Path("/app/ongiini").is_dir():
    sys.path.insert(0, "/app")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ongiini import contributions, ratings  # noqa: E402

BASE_URL = "https://ongiini.ai/rate/"


def review_answers(con, where: str = "1", params: tuple = ()) -> int:
    """Review answers a destructive command would delete — those must never go by accident."""
    return con.execute("SELECT COUNT(*) FROM ratings r JOIN items i USING (item_key) "
                       f"WHERE i.grp = 'review' AND {where}", params).fetchone()[0]


def refuse(n: int) -> int:
    print(f"REFUSED: this would delete {n} review answers. Back them up first "
          "(scripts/backup_ratings.sh) and re-run with --delete-review-answers.")
    return 1


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    con = ratings.connect()
    cmd, args = argv[0], argv[1:]
    force = "--delete-review-answers" in args
    args = [a for a in args if a != "--delete-review-answers"]
    if cmd == "add-rater" and len(args) >= 2:
        dialects = args[1:]
        bad = [d for d in dialects if d not in ("oshindonga", "oshikwanyama")]
        if bad:
            print(f"unknown dialect(s): {bad}")
            return 1
        token = ratings.add_rater(con, args[0], dialects)
        print(f"{args[0]} ({', '.join(dialects)}):\n  {BASE_URL}#t={token}")
    elif cmd == "add-reviewer" and len(args) == 3:
        try:
            token = ratings.add_reviewer(con, args[0], args[1], args[2])
        except ValueError as exc:
            print(exc)
            return 1
        print(f"{args[0]} (review {args[2]}, {args[1]}):\n  {BASE_URL}#t={token}")
    elif cmd == "open-rating" and args in (["on"], ["off"]):
        ratings.set_open_rating(con, args[0] == "on")
        print(f"open rating {args[0]} (reviewers are not affected)")
    elif cmd == "review-progress" and len(args) == 1:
        for r in con.execute(
                "SELECT ra.rater_id, ra.label, ra.dialects, COUNT(r.rating_id) done, "
                "(SELECT COUNT(*) FROM items i WHERE i.round = ra.review_round AND i.grp = 'review' "
                " AND i.dialect = ra.dialects) total, "
                "SUM(r.verdict = 'good') good, SUM(r.verdict = 'almost') almost, SUM(r.verdict = 'wrong') wrong, "
                "SUM(r.verdict = 'cant_judge') cant, datetime(MAX(r.created_at), 'unixepoch') last "
                "FROM raters ra LEFT JOIN ratings r USING (rater_id) WHERE ra.review_round = ? "
                "GROUP BY ra.rater_id", (args[0],)):
            print(dict(r))
    elif cmd == "load" and len(args) == 1:
        n = ratings.load_items(con, json.loads(Path(args[0]).read_text()))
        print(f"{n} items in the database")
    elif cmd == "report":
        print(json.dumps(ratings.summary(con), indent=2))
    elif cmd == "raters":
        for r in con.execute("SELECT rater_id, label, dialects, first_language, home_dialect, active, blocked, "
                             "CASE WHEN contributor_hash IS NULL THEN 'admin' ELSE 'whatsapp' END via, "
                             "(SELECT COUNT(*) FROM ratings x WHERE x.rater_id = raters.rater_id) n FROM raters"):
            print(dict(r))
    elif cmd == "block" and len(args) == 1:
        with con:
            con.execute("UPDATE raters SET blocked = 1 WHERE rater_id = ?", (int(args[0]),))
        print("blocked")
    elif cmd == "block-number" and len(args) == 1:
        ratings.block_contributor(con, contributions.hash_msisdn(args[0].lstrip("+").replace(" ", "")))
        print("blocked (only the salted hash is stored)")
    elif cmd == "deactivate" and len(args) == 1:
        with con:
            con.execute("UPDATE raters SET active = 0 WHERE rater_id = ?", (int(args[0]),))
        print("deactivated")
    elif cmd == "delete-rater" and len(args) == 1:
        if not force and (n := review_answers(con, "r.rater_id = ?", (int(args[0]),))):
            return refuse(n)
        ratings.delete_rater(con, int(args[0]))
        print("deleted rater and their ratings")
    elif cmd == "delete-round" and len(args) == 1:
        if not force and (n := review_answers(con, "i.round = ?", (args[0],))):
            return refuse(n)
        print(f"{ratings.delete_round(con, args[0])} items deleted (with their ratings)")
    elif cmd == "reset-test" and args == ["--yes"]:
        if not force and (n := review_answers(con)):
            return refuse(n)
        with con:
            for t in ("ratings", "items", "raters"):
                con.execute(f"DELETE FROM {t}")
        print("all raters, items and ratings deleted")
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
