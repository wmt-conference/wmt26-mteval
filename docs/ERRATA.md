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
252,923 official rows (`collection = 'official'`) and 110,319 challenge rows, 363,242 in all. The
paper's own total of 363,242 is consistent only with 252,923.

**Release.** The validation gates compare the official items and pairs of every language pair with
Table 2, so the release follows Table 2 and the data: 252,923 official pairs.

### 2. Translations per source segment

**Paper.** Section 3.1 says that participants judged "up to 47 different translations per source
segment, and in no case fewer than 23".

**Data.** Every official item has between 27 and 31 candidate translations (`items.n_systems`,
which counts the MT systems and the references that are among the candidates). The data lie
inside the paper's range, but the range is much wider than what the data show. The paper does not
say what accounts for the difference. Challenge items have one or two candidates, apart from the
RefQuality items, which have 29 or 30.

**Release.** `items.n_systems` and the rows of `pairs` are the counts to use.

### 3. "18 direct Task 3 submissions"

**Paper.** Section 7.2 says that Table 10 "reports rankings for the 18 direct Task 3 submissions"
and the converted organiser Task 1 baselines.

**Data.** Table 10 has 18 rows: 9 direct Task 3 submissions and 9 baselines (converted from Task 1).
The data hold 13 direct Task 3 runs. The four that Table 10 leaves out do not cover all eight
priority pairs, as its caption says: the two Lexicala-QE-Ensemble runs (6 of 8 pairs) and the two
PTT-MULTITAN-v2 runs (no priority pair).

**Release.** `systems` lists all 13 runs (`tasks = ['task3']`, `is_baseline = false`), and
`covers_all_priority` is true for the nine that Table 10 ranks and false for the four it omits.

### 4. Two descriptions of the deduplication of baseline error spans

**Paper.** Section 4.2 says that exact duplicates were removed "matching on start and end offset,
keeping the major-severity annotation when the severities disagreed". Section 6.3.1 says that the
deduplicated variants were deduplicated "by identical start offset, end offset, severity, and
category". The two rules differ. The paper does not say which snapshot its baseline rows use;
Table 9 lists both, as `[no dedup]` and `[dedup]`, and shows the four xCOMET pairs once, because
they are identical before and after deduplication.

**Data.** The `provenance.json` that the organisers supplied with their deduplicated files records
the rule of Section 4.2 (start and end offset, major severity kept). That file is not part of
this release.

**Release.** `baselines.variant = 'official'` is the deduplicated snapshot of the LLM baselines
(and the xCOMET files, which are identical) and `'alt'` is the original, not deduplicated
snapshot, with Task 1 columns only. The baseline runs from before the deduplication fix are not
in the typed configs.

### 5. Item counts of two challenge sets

**Paper.** Table 14 and Section 8.2 give 2,886 items for QEbreak and 171 for LIGATUR. The item
counts in the header of Table 15, which add up per language pair, give 1,749 for QEbreak and 159
for LIGATUR.

**Data.** The release holds 2,886 QEbreak and 171 LIGATUR items, as Table 14 does. The paper does not
say why Table 15 lists fewer; it may show only the items that were scored.

| set | language pair | Table 15 | data |
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

**Paper.** The same baseline carries different names in different tables. For example, the Gemma 4
baseline is "Gemma 4 - thinking" in Table 4, "Gemma 4 - Reasoning" in Table 8 and "Gemma 4
Reasoning" in Table 10, and the GPT-5.6 Sol baseline is "GPT-5.6 Sol - medium thinking" in
Table 4, "GPT-5.6 Sol - Medium Reasoning" in Table 8 and "GPT-5.6 Sol Medium Reasoning" in
Table 10.

**Release.** `systems.canonical_name` gives each run one name. A participant run has the name it
carries in the ranking table of its own task (Table 4, 8 or 10), or else in Table 3; a baseline
has its name in Table 8, and the xCOMET baselines that differ only in their use of a reference
are told apart as "(reference-free)" and "(reference-based)".
[`data/metadata/run_names.tsv`](../data/metadata/run_names.tsv) lists every name, the model
behind it and its citation, run by run.
