# Errata: where the findings paper and the data differ

Applies to release v0.9.

This file lists the places where the findings paper (Lavie et al., 2026, *Proceedings of the
Eleventh Conference on Machine Translation (WMT)*, pages 931-1030) and the released data differ.
For each one it gives what the paper says, what the data show and what the release does. Where
the paper gives no reason for a difference, we say so rather than guess. Section and table
numbers are those of the paper.

### 1. Number of official segment pairs

**Paper.** Section 3.1 says that the official evaluation set "comprised 252,919 segment pairs" and
that, with the challenge sets, the total was 363,242.

**Data.** The "# Pairs in Test" column of Table 2 adds up to 252,923, and the `pairs` config holds
252,923 official rows (`collection = 'official'`) and 110,319 challenge rows, 363,242 in all. Given
the 110,319 challenge rows, the paper's total of 363,242 implies 252,923 official pairs, not
252,919; the paper itself gives no total for the challenge pairs.

**Release.** The validation gates compare the official items and pairs of every language pair with
Table 2, so the release follows Table 2 and the data: 252,923 official pairs.

### 2. Translations per source segment

**Paper.** Section 3.1 says that participants judged "up to 47 different translations per source
segment, and in no case fewer than 23".

**Data.** Every official item has between 27 and 31 candidate translations (`items.n_systems`,
which counts the MT systems and the references that are among the candidates), and so does every
official language pair. Across all 21 official language pairs there are 47 distinct candidates:
44 MT systems and three kinds of reference (from scratch, post-edited and pseudo-reference). The
paper does not say what accounts for the difference between its range and the 27 to 31 per item.
Challenge items have one or two candidates, apart from the RefQuality items, which have 29 or 30.

**Release.** `items.n_systems` and the rows of `pairs` are the counts to use.

### 3. "18 direct Task 3 submissions"

**Paper.** Section 7.2 says that Table 10 "reports rankings for the 18 direct Task 3 submissions"
and the converted organiser Task 1 baselines.

**Data.** Table 10 has 18 rows: 9 direct Task 3 submissions and 9 baselines (converted from Task 1),
so 18 is the number of rows, not of direct submissions. The data hold 13 direct Task 3 runs. The
four that Table 10 leaves out do not cover all eight priority pairs, as its caption says: the two
Lexicala-QE-Ensemble runs (6 of 8 pairs) and the two PTT-MULTITAN-v2 runs (no priority pair; the
caption names PTT-MULTITAN-v2 once, and the data hold two runs of it).

**Release.** `systems` lists all 13 runs (`tasks = ['task3']`, `is_baseline = false`), and
`covers_all_priority` is true for the nine that Table 10 ranks and false for the four it omits.

### 4. Two descriptions of the deduplication of baseline error spans

**Paper.** Section 4.2 says that exact duplicates were removed "matching on start and end offset,
keeping the major-severity annotation when the severities disagreed". Section 6.3.1 says that the
deduplicated variants were deduplicated "by identical start offset, end offset, severity, and
category". The two rules differ. The paper is also not consistent about which snapshot its baseline
rows use: Section 4.2 presents the deduplication as part of how the baselines were made, which
implies that every baseline row is deduplicated, while Section 6.3.1 calls the non-deduplicated
outputs "the original baseline outputs" and the deduplicated ones variants. No table except Table 9
labels its baseline rows. Table 9 lists both snapshots, as `[no dedup]` and `[dedup]`, and shows the
four xCOMET pairs once, because they are identical before and after deduplication.

**Data.** The `provenance.json` that the organisers supplied with their deduplicated files records
the rule of Section 4.2 (one error per start and end offset, preferring a major error), and settles
what the paper leaves open: when several errors share the offsets and severity does not decide, the
first is kept. The data agree: no `official` baseline prediction repeats a start and end offset, and
for each LLM baseline the number of spans removed between `alt` and `official` equals the number
that `provenance.json` records. That file is not part of this release.

**Release.** `baselines.variant = 'official'` holds the deduplicated snapshot of the 12 LLM
baselines, the four xCOMET baselines (whose files are identical before and after deduplication) and
the seven segment-level metric baselines (BLEU, spBLEU, chrF, BERTScore, COMET22, COMETKiwi22 and
YiSi-1), which have a single snapshot. `'alt'` holds the original, not deduplicated snapshot of the
12 LLM baselines, with Task 1 columns only. An earlier snapshot of 15 baselines, which the
organisers label `pre_dedup_pre_enhy_fix` (taken before the deduplication and before a second fix
that the label calls `enhy_fix`), is not in the typed configs; it remains in the `grand` config. The
`official` label is provisional: we chose the deduplicated snapshot because Section 4.2 describes
deduplication as part of how the baselines were made. Deduplication changes neither the Task 2
scores that the baselines predict directly (Table 8) nor whether a segment has any error at all,
which is all that the conversion to Task 3 uses (Tables 10-13 and 65-85); in the organisers' files,
no Task 2 score and no error-free label differs between the two snapshots. It does change everything
computed from the spans themselves: the scores converted to Task 2, which Table 9 shows for both
snapshots, and the Task 1 results in Tables 4-7, Tables 16-36 and Figures 1-3. Of these, the error
counts of Table 6 can be checked without the human annotations, and they fit the deduplicated
snapshot: averaged over the eight priority pairs (official items), the reference-free Qwen 3.6
baseline with reasoning (run `qwen`) predicts 3.21 errors per prediction in `official` and 5.16 in
`alt`, against 3.11 in Table 6, and the reference-based one (run `qwen_ref`) 4.08 and 6.39, against
4.10. The paper does not say which rows Table 6 covers, so these figures are close rather than
equal, and the Gemma 4 baselines change too little to tell. The rest will be checked against the
paper in v1.0, once the human annotations are part of the release.

### 5. Item counts of two challenge sets

**Paper.** Table 14 gives 2,886 items for QEbreak and 171 for LIGATUR, and Section 8.2 gives the
same figures (2,886 "segments" for QEbreak; a "171-item" set for LIGATUR). The per-language-pair
item counts in the header row of Table 15 add up to 1,749 for QEbreak and 159 for LIGATUR.

**Data.** The release holds 2,886 QEbreak and 171 LIGATUR items, as Table 14 does. The paper does
not say why Table 15 lists fewer.

| set | language pair | Table 15 | items in data |
| --- | --- | ---: | ---: |
| QEbreak | `ces-deu` | 108 | 216 |
| QEbreak | `ces-ukr` | 252 | 417 |
| QEbreak | `eng-arb` (`eng-ara` in Table 15) | 88 | 176 |
| QEbreak | `eng-ces` | 251 | 429 |
| QEbreak | `eng-deu` | 607 | 894 |
| QEbreak | `eng-jpn` | 31 | 51 |
| QEbreak | `eng-kor` | 33 | 66 |
| QEbreak | `eng-rus` | 231 | 363 |
| QEbreak | `eng-zho_CN` (`eng-zho` in Table 15) | 138 | 254 |
| QEbreak | `zho-jpn` | 10 | 20 |
| LIGATUR | `eng-deu` | 89 | 95 |
| LIGATUR | `eng-hin` | 70 | 76 |

**Release.** The `challenge_sets` config counts the items and pairs actually present in `pairs`.

### 6. Baseline names drift between tables

**Paper.** The same baseline carries different names in different tables. For example, the
reference-free Gemma 4 baseline with reasoning (run `gemma`) is "Gemma 4 - thinking" in Table 4,
where the reference-based variant carries the same name and differs only by the missing asterisk,
"Gemma 4 - Reasoning" in Table 8 and "Gemma 4 Reasoning" in Table 10; the GPT-5.6 Sol baseline at
medium reasoning is "GPT-5.6 Sol - medium thinking" in Table 4, "GPT-5.6 Sol - Medium Reasoning" in
Table 8 and "GPT-5.6 Sol Medium Reasoning" in Table 10.

**Release.** `systems.canonical_name` gives each run one name. A participant run has the name it
carries in the ranking table of its own task (Table 4, 8 or 10), or else in Table 3; where Table 10
adds a run number in brackets, as in "FACET [869556]", the canonical name leaves it out. A baseline
has its name in Table 8, and the xCOMET baselines that differ only in their use of a reference are
told apart as "(reference-free)" and "(reference-based)". The one baseline that Table 8 does not
list, GPT-5.6 Sol at xHigh reasoning (run `gpt56_sol_xhigh_no_ref`), is "GPT-5.6 Sol - xhigh" in
Table 7 and "GPT-5.6 Sol - xhigh thinking" in Table 16; the release names it "GPT-5.6 Sol - Xhigh
Reasoning", after the pattern of Table 8.
[`data/metadata/run_names.tsv`](../data/metadata/run_names.tsv) lists every name, the model
behind it and its citation, run by run.

### 7. The GPT-5.6 Sol baseline at xHigh reasoning

**Paper.** Section 4.2.1 says that the GPT-5.6 Sol model at xHigh reasoning was "able to complete
6 of the 8 pairs within timing constraints". Neither Table 4 nor Table 8 lists it, and of the
per-direction Task 1 tables only Table 16 (`ces-deu`) does. Table 7 compares it
with the medium-reasoning run only on the translations for which both completed their reasoning.
The paper does not say which two pairs were left incomplete or how their predictions were filled.

**Data.** The run (`gpt56_sol_xhigh_no_ref`) has rows for all eight priority pairs, 95,323 in all.
Its `t2_score` is 50.0 in every row. In `eng-sme`, 13,087 of the 13,311 rows (98.3%) have no
error and no omission, and in `eng-ind` 6,084 of 10,261 (59.3%), against 16.5% to 26.8% in its
other six pairs; the run at medium reasoning has 11.1% in `eng-sme` and 20.2% in `eng-ind`. None
of the 13,311 `eng-sme` rows has an error span; the other 224 report only an omission.

**Release.** The values are the organisers' and are copied unchanged. Do not use the `t2_score` of
this run, and treat its Task 1 predictions for `eng-sme` and `eng-ind` as incomplete. Its
`covers_all_priority` is true, because it has rows for all eight pairs.
