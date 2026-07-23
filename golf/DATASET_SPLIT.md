# Golf dataset — train / val / test split (authoritative)

Source corpus: **`~/2026/dataset/golf/`** on the offline box (SRA0402), organized `{Indoor,Outdoor}/<Person>/`
with a video + audio + IMU per capture. **27 people · 931 videos · 350 min** of footage (clips are short,
~22 s each — so **video COUNT badly misrepresents footage; we split by DURATION**).

Frozen **2026-07-23**. The machine-readable manifest is `golf_split_manifest.csv` (person → split);
reproduce it with the command at the bottom. Everything downstream MUST honor this split.

## Split rules (why it's trustworthy)
- **Whole PERSON is the atomic unit** — every clip of a person lands in one split (no identity leakage).
  Same name across Indoor+Outdoor (e.g. Jonathan) is merged and assigned once.
- **Balanced by MINUTES, not video count** — a person with 85 short clips (~8 min) carries less footage
  than one with 7 long clips (~17 min); counts would mislead. Probed via cv2 header (cached).
- **Both domains in every split** — Indoor and Outdoor stratified independently.
- **Indoor footage all contains holes** → every Indoor person in val/test covers the `hole` class
  (~24–25 min of hole footage in each of val and test).
- **Jonathan** (biggest, 32.7 min, both-domain) and **Friend Capture** (mixed-identity folder) → **train**.
- **val = model selection / early-stop. test = touched once at the end.** Both are FROZEN.

## Result

| split | people | **minutes** | **min share** | videos | video share | Indoor / Outdoor min |
|---|---|---|---|---|---|---|
| **train** | 17 | **239.1** | **68%** | 474 | 51% | 156.3 / 82.8 * |
| **val**   | 5  | **54.9**  | **16%** | 216 | 23% | 24.1 / 30.8 |
| **test**  | 5  | **55.5**  | **16%** | 241 | 26% | 25.0 / 30.5 |

> Note the divergence: by **minutes** it's a clean 68 / 16 / 16, but by **video count** val/test look
> oversized (23% / 26%) — proof that duration is the right metric here.
>
> \* Train Indoor/Outdoor split Jonathan's dual-domain footage into its real parts
> (**11.0 min Indoor + 21.7 min Outdoor** = 32.7), so every row's Indoor + Outdoor = its total
> (156.3 + 82.8 = 239.1). val/test have no dual-domain person.
> Hole (=Indoor) coverage: train 156 / val 24 / test 25 min.

- **val**  people: Michael, Joy, Hiro, Ramu, Aryan  (Indoor: Hiro, Ramu, Aryan · Outdoor: Michael, Joy)
- **test** people: Yujin, Alex, Kun, Madhu, AJ      (Indoor: Kun, Madhu, AJ · Outdoor: Yujin, Alex)

## Per-person assignment (sorted by minutes)

| person | domain | videos | minutes | split |
|---|---|---:|---:|---|
| Jonathan | Indoor+Outdoor | 129 | 32.7 | train |
| Marx | Outdoor | 67 | 30.9 | train |
| Michael | Outdoor | 36 | 27.8 | **val** |
| Yujin | Outdoor | 62 | 24.6 | **test** |
| Yohan | Outdoor | 28 | 17.9 | train |
| Eugene | Indoor | 7 | 17.0 | train |
| Edgar | Indoor | 7 | 15.2 | train |
| Pratik | Indoor | 6 | 14.4 | train |
| Arvind | Indoor | 6 | 13.9 | train |
| Sunith | Indoor | 6 | 12.5 | train |
| Venki | Indoor | 6 | 12.4 | train |
| David | Indoor | 6 | 12.3 | train |
| Akash | Indoor | 6 | 10.9 | train |
| Simon | Indoor | 29 | 10.8 | train |
| Hiro | Indoor | 6 | 10.0 | **val** |
| Vaibhav | Indoor | 85 | 9.3 | train |
| Mallikarjun | Indoor | 6 | 8.6 | train |
| AJ | Indoor | 6 | 8.6 | **test** |
| Friend Capture | Outdoor | 47 | 8.6 | train |
| Kun | Indoor | 85 | 8.5 | **test** |
| Vivek | Indoor | 22 | 8.0 | train |
| Ramu | Indoor | 84 | 7.9 | **val** |
| Madhu | Indoor | 85 | 7.9 | **test** |
| Aryan | Indoor | 84 | 6.2 | **val** |
| Alex | Outdoor | 3 | 5.9 | **test** |
| Joonhee | Outdoor | 11 | 3.7 | train |
| Joy | Outdoor | 6 | 3.0 | **val** |

## Reproduce
```bash
~/ml-models/.venv/bin/python golf/split_golf_dataset.py ~/2026/dataset/golf --by minutes \
    --val-frac 0 --test-frac 0 \
    --pin-val  "Michael,Joy,Hiro,Ramu,Aryan" \
    --pin-test "Yujin,Alex,Kun,Madhu,AJ" \
    --out golf_split_manifest.csv
```
`--val-frac 0 --test-frac 0` = no auto-fill; eval = exactly the pinned people, everyone else → train.
Durations are cached in `golf_split_manifest_durations.json` (re-runs are instant).

## Downstream rules (do not break the split)
- **Freeze val/test**: never mine / review / train on those 10 people. Feed the **train** people to
  `select_review_frames.py --exclude-trained` (and any mining) so held-out footage never leaks in.
- **Label val/test by hand** as ground truth (especially holes); train people may be auto-mined/reviewed.
- If new people/footage arrive, re-run the split (or pin the newcomers) and update this doc — keep the
  minutes/people/ratio table current.
