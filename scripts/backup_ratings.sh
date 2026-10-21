#!/usr/bin/env bash
# Snapshot the rating database (ongiini.ai/rate/) every 10 minutes (cron).
#
# The reference review (round v1_review, two invited native speakers, ~1200
# answers) must never be lost, and the Spark has one disk. This makes a
# consistent online copy (sqlite .backup, safe while the webhook writes),
# checks it, and keeps timestamped gzip snapshots plus a readable JSONL of
# the review answers OUTSIDE the repo (a git clean or a broken data/ mount
# can't touch them). The Mac pulls the folder off the machine
# (deploy/mac/com.ongiini.ratings-backup.plist).
#
# Install (crontab -e on the Spark, as the repo owner):
#   */10 * * * * bash ~/dev/Ongiini/scripts/backup_ratings.sh
#
# Alerts (journal + operator WhatsApp via deploy/train/notify.sh) when the
# copy fails its integrity check or the number of answers SHRANK since the
# last snapshot — then the previous "latest" files are kept untouched.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB="${ONGIINI_RATINGS_DB:-$REPO_DIR/data/ratings.sqlite}"
OUT="${ONGIINI_RATINGS_BACKUP_DIR:-$HOME/backups/ratings}"
KEEP_DAYS=60
mkdir -p "$OUT/snapshots"

alert() {
  logger -t ongiini-ratings-backup "$*"
  echo "$(date '+%F %T') ALERT $*" >> "$OUT/backup.log"
  bash "$REPO_DIR/deploy/train/notify.sh" "Ratings-Backup: $*" >/dev/null 2>&1 || true
}

[ -s "$DB" ] || { alert "database missing or empty: $DB"; exit 1; }
stamp="$(date '+%Y%m%d-%H%M')"
tmp="$OUT/.tmp-$stamp.sqlite"
if ! sqlite3 "$DB" ".timeout 10000" ".backup '$tmp'"; then
  rm -f "$tmp"; alert "sqlite backup failed"; exit 1
fi
if [ "$(sqlite3 "$tmp" 'PRAGMA integrity_check;')" != "ok" ]; then
  rm -f "$tmp"; alert "integrity check failed on the copy"; exit 1
fi

n_all="$(sqlite3 "$tmp" 'SELECT COUNT(*) FROM ratings;')"
n_review="$(sqlite3 "$tmp" "SELECT COUNT(*) FROM ratings r JOIN items i USING (item_key) WHERE i.grp = 'review';")"
prev_all="$(cut -d' ' -f1 "$OUT/latest.count" 2>/dev/null || echo 0)"
prev_review="$(cut -d' ' -f2 "$OUT/latest.count" 2>/dev/null || echo 0)"

gzip -c "$tmp" > "$OUT/snapshots/ratings-$stamp.sqlite.gz"
if [ "$n_all" -lt "$prev_all" ] || [ "$n_review" -lt "$prev_review" ]; then
  # A reviewer's undo removes one answer: only alert on a real drop.
  if [ $((prev_review - n_review)) -gt 2 ] || [ $((prev_all - n_all)) -gt 2 ]; then
    rm -f "$tmp"
    alert "answers shrank: all $prev_all → $n_all, review $prev_review → $n_review (snapshot kept, latest untouched)"
    exit 1
  fi
fi

mv "$tmp" "$OUT/latest.sqlite"
sqlite3 -json "$OUT/latest.sqlite" "
  SELECT ra.label AS reviewer, i.sentence_id, i.dialect, i.english, i.text AS reference, r.verdict,
         r.issues, r.suggestion, r.cant_reason, r.duration_ms, r.seq, datetime(r.created_at, 'unixepoch') AS at
  FROM ratings r JOIN items i USING (item_key) JOIN raters ra USING (rater_id)
  WHERE i.grp = 'review' ORDER BY ra.rater_id, r.seq;" > "$OUT/review_answers-latest.json.tmp" \
  && mv "$OUT/review_answers-latest.json.tmp" "$OUT/review_answers-latest.json"
echo "$n_all $n_review" > "$OUT/latest.count"
echo "$(date '+%F %T') ok all=$n_all review=$n_review" >> "$OUT/backup.log"
find "$OUT/snapshots" -name 'ratings-*.sqlite.gz' -mtime +"$KEEP_DAYS" -delete
