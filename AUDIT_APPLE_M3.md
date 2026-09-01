# Audit: "Apple M3" example provenance (Reviewer 1, Round 2)

A reviewer asked us to explain the provenance of an AG News example referencing an
"Apple M3 announcement" that they recalled from an earlier manuscript draft. This
document records the exact audit we ran in response, so it is independently
checkable rather than taken on our word.

## Command 1 — search every released per-sample CSV (all 3 datasets, both models, all k) for "M3"

```
grep -il "m3" vote_records/reviewer_data_package/per_sample_vote_count_records_scored/*.csv
```

Result: **no files matched** (18 files searched).

## Command 2 — search all 6 AG News CSVs for "apple" with context

```
for f in vote_records/.../ag_news*.csv; do
  grep -i -o '.\{0,15\}apple.\{0,70\}' "$f" | sort -u
done
```

Result: 13 distinct matching records across the 6 files, all 2004-era Apple Computer
Inc. stories (iPod, iTunes Music Store, PowerBook battery recalls, iMac launch,
customer-satisfaction rankings). None mentions any chip, processor, or "M3" of any
kind.

## Command 3 — dataset vintage check

AG News (Zhang, Zhao & LeCun, NeurIPS 2015) is built from the AG corpus of news
articles gathered by ComeToMyHead in 2004. The Apple M3 chip was announced in 2023,
roughly two decades later. A genuine AG News record referencing "Apple M3" is not
possible.

## Command 4 — manuscript check

Every table cell in the current manuscript was checked programmatically
(case-insensitive) for "apple" or "m3". No match.

## Conclusion

We could not locate this example in the released data, the current manuscript, or
our own drafting history. We do not believe it originates from real data in this
study. If it appeared in an intermediate manuscript draft, we were not able to
identify which one from our records alone.
