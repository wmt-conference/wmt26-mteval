---
pretty_name: WMT26 MT-Eval
task_categories:
- translation
language:
{{language_list}}
size_categories:
- {{size_category}}
tags:
- machine-translation
- mt-evaluation
- quality-estimation
- wmt
configs:
- config_name: items
  data_files:
  - split: evaluation
    path: data/items/evaluation-*.parquet
- config_name: pairs
  default: true
  data_files:
  - split: evaluation
    path: data/pairs/evaluation-*.parquet
- config_name: submissions
  data_files:
  - split: evaluation
    path: data/submissions/evaluation-*.parquet
- config_name: baselines
  data_files:
  - split: evaluation
    path: data/baselines/evaluation-*.parquet
- config_name: systems
  data_files:
  - split: evaluation
    path: data/systems/evaluation-*.parquet
- config_name: mt_systems
  data_files:
  - split: evaluation
    path: data/mt_systems/evaluation-*.parquet
- config_name: challenge_sets
  data_files:
  - split: evaluation
    path: data/challenge_sets/evaluation-*.parquet
- config_name: grand
  data_files:
  - split: evaluation
    path: wmt26-grand.parquet
---

# WMT26 MT-Eval

Test data and evaluator predictions of the **WMT26 Shared Task on Automated Translation Quality
Evaluation Systems**, the shared task on MT metrics and quality estimation (findings paper: Lavie
et al., 2026; see [Citation](#citation)).

Participants built automatic evaluators of translation quality. The organisers gave them one test
set, made of source texts and candidate translations taken from the WMT26 General Machine
Translation task (General MT), and asked for contributions of four kinds. Tasks 1 to 3 asked for
predictions; Task 4 asked for challenge sets:

| task | what it asks for | columns |
| --- | --- | --- |
| Task 1 | error spans with a severity, and omissions | `t1_errors`, `t1_omission` |
| Task 2 | a quality score for each translation, from 0 to 100 | `t2_score` |
| Task 3 | whether a translation is free of errors | `t3_error_free` |
| Task 4 | challenge sets that probe the evaluators: five were submitted by research teams, and the two RefQuality sets are the organisers' own | `challenge_sets`, `challenge_set` |

The test set covers {{n_official_lps}} official language pairs, {{n_priority_lps}} of them
priority pairs that the organisers recommended to participants who could not run the whole set,
and {{n_challenge_sets}} challenge sets. Submissions to one task were also converted
automatically to the downstream tasks in the findings paper; those conversions are not part of
v0.9.

This repository holds the test set, every candidate translation, and the predictions of every
participant run and every organiser baseline, as typed Parquet tables that load in one line.

{{counts}}

## What this release contains

This is **v0.9**: typed data configs and documentation.

- It contains the seven typed configs below and the organisers' original single file as `grand`.
- v0.9 contains no human-annotation gold in its typed configs. The human evaluation was carried
  out by General MT and is public in
  [its release](https://github.com/wmt-conference/wmt26-general-mt/releases/tag/humeval).
  A gold config is planned for v1.0, together with the converted predictions, the results tables
  and the scoring code.
  The unchanged `grand` config still carries the organisers' raw annotation records in its
  `ground_truth__record` column; see [Human annotations](#human-annotations).

## Quick start

Every config has one split, named `evaluation`: this is one evaluation set, not partitioned into
train/test sets. The typed configs load with `datasets>=2.18`:

```python
from datasets import load_dataset

pairs = load_dataset("wmt/wmt26-mteval-data", "pairs", split="evaluation")
print(pairs[0]["lp"], pairs[0]["src"], pairs[0]["hyp"], sep="\n")
```

`pairs` is the default config. `submissions` and `baselines` are large ({{n_submission_rows}}
and {{n_baseline_rows}} rows), so you may prefer to stream them or to download one config:

```python
subs = load_dataset("wmt/wmt26-mteval-data", "submissions", split="evaluation", streaming=True)

base = load_dataset("wmt/wmt26-mteval-data", "baselines", split="evaluation", streaming=True)
# `baselines` also holds the original, pre-deduplication snapshot (`alt`); see Warnings
base = base.filter(lambda row: row["variant"] == "official")
```

```bash
# scripts/download.py is in the GitHub repository, https://github.com/wmt-conference/wmt26-mteval
uv run python scripts/download.py --config submissions --out data/hf
```

The Parquet files can also be queried in place, for example with DuckDB:

```python
import duckdb

base = "hf://datasets/wmt/wmt26-mteval-data/data"
duckdb.sql(f"""
    SELECT lp, system, count(*) AS n, sum(hyp_is_empty::int) AS n_empty
    FROM '{base}/pairs/*.parquet'
    WHERE collection = 'official'
    GROUP BY lp, system ORDER BY lp, system
""").show()
duckdb.sql(f"SELECT run_id, name, tasks, n_items FROM '{base}/systems/*.parquet'").show()
```

The `grand` config is the organisers' single file with JSON columns. The `datasets` library reads
those as dicts from version 4.7.0 on (`pip install "datasets>=4.7.0"`); older versions cannot
load this config, but they load the seven typed configs:

```python
grand = load_dataset("wmt/wmt26-mteval-data", "grand", split="evaluation")
```

## Configs

{{config_table}}

- Join the hypotheses to the predictions on `(item_id, system)`, then join the predictions to
  `systems` on `run_id`; for `baselines`, filter `variant == 'official'` first (see Warnings).
- `baselines.variant` is `official` for the released predictions: the deduplicated snapshot of
  the LLM baselines, the xCOMET baselines (whose files are identical before and after
  deduplication) and the segment-level metric baselines, which have a single snapshot. `alt` is
  the original, not deduplicated snapshot of the LLM baselines, with Task 1 columns only. An
  earlier snapshot, which the organisers label `pre_dedup_pre_enhy_fix`, is in the `grand` config
  only. The `official` label is provisional: Section 4.2 of the findings paper describes
  deduplication as part of how the baselines were made, so the deduplicated snapshot is `official`
  for now. It will be checked against the paper in v1.0. Only the results computed from the
  error spans can tell the two snapshots apart (Tables 4-7 and 16-36 of the findings paper), and
  Table 6, which needs no human annotation, already fits the deduplicated snapshot: its errors per
  prediction for the Qwen 3.6 baselines match the deduplicated predictions, not the original ones.
- A run may cover only some items, and then has no rows for the others; `systems.n_items`, `lps`
  and `covers_all_priority` say what it covers.
- `systems` holds allow-listed model-card answers only: a fixed list of fields was copied, and
  nothing else was read. The links in `link` were volunteered by the teams for publication.
  `name` is the name the team gave its run (followed by the run id where several runs share one
  name), and `canonical_name` is the name the findings paper uses.
- `t1_instruction_fault` is null in every row of this release.
- The baseline diagnostic `stage1_finish_reason`, which some baseline runs record, appears only
  in the `grand` config.

## How to read the data

{{conventions}}

## Warnings

- **`baselines` holds two snapshots of some runs.** {{n_alt_baseline_runs}} runs have an `alt`
  snapshot next to their `official` one, and its {{n_alt_baseline_rows}} `alt` rows share their
  `run_id` with the official rows. Counting or averaging `baselines` by `run_id`, or joining it to
  `systems` on `run_id`, therefore counts those runs twice. Filter `variant == 'official'` unless
  you want the pre-deduplication snapshot.
- **References are among the candidate translations of official items.** In the official set,
  every available reference is also a hypothesis (`system_type` is `human_scratch`,
  `human_postedit` or `pseudo_ref`). Challenge hypotheses can equal the reference too:
  `challenge_hyp` rows can be `is_reference_candidate = true`. In all, {{n_reference_candidates}}
  pairs have a hypothesis that equals the item's reference, up to surrounding whitespace, and
  carry `is_reference_candidate = true`. A reference-based evaluator is given the reference as
  well, so it sees a copy of it on those rows.

  `is_reference_candidate` also flags MT outputs that are identical to the reference, so it does
  not mark the reference systems only. By `system_type`, the flagged pairs are:

{{reference_candidate_breakdown}}

  To drop the reference itself, filter on `system_type`: remove the flagged rows of
  `human_scratch`, `human_postedit` and `pseudo_ref`. Do not drop every `is_reference_candidate`
  row: that removes real MT outputs and challenge hypotheses as well.
- **The GPT-5.6 Sol baseline at xHigh reasoning is incomplete.** The findings paper says this
  run completed only 6 of the 8 priority pairs. In the data, its `t2_score` is the same
  placeholder value, 50.0, in every row, and its Task 1 predictions are empty for almost all of
  `eng-sme` (no row has an error span) and for most of `eng-ind`. Do not use its `t2_score`, and
  treat its `eng-sme` and `eng-ind` predictions as incomplete.
- **Some hypotheses are empty.** There are {{n_hyp_empty}} empty or whitespace-only
  hypotheses ({{n_hyp_empty_exact}} exactly empty and {{n_hyp_whitespace_only}}
  whitespace-only); `hyp_is_empty` is true for all of them. They are kept, and so are the
  predictions made for them.
- **RefQuality items repeat sources of official items.** The two RefQuality challenge sets
  (`RefQuality-PD` and `RefQuality-PE`) re-use the sources of official items with a different
  reference, a pseudo-reference or a post-edit, to show what the reference type changes. Their
  ids differ from the official ones, so filter on `collection` before you count sources. A
  `doc_id` that starts with `RefQuality-PD_RefQuality-PE_` marks the pseudo-reference variant of
  an item whose alternative reference is a post-edit.
- **Three language pairs have little human annotation.** The share of pairs that General MT
  annotated is only 10.6 % for `ces-vie`, 17.1 % for `eng-zho_CN` and 19.0 % for `eng-sme`
  (findings paper, Table 2), so meta-evaluation on them rests on few judgements.
- **Items are paragraph-level.** The last field of `item_id` is a paragraph index, and the
  "segments" of the findings paper are these items. A hypothesis can span several sentences, and
  the `start` and `end` of a Task 1 span are character offsets into that whole hypothesis.
- **`Cohere CAT+` is two things.** It is an MT system that has hypotheses in `pairs` and an
  evaluator that has a row in `systems`. Do not join `pairs.system` to `systems.name`: in the
  predictions, `run_id` names the evaluator and `system` the translation it judged.
- **Task 2 scores can fall outside 0-100.** Some runs predict below 0 or above 100, and some
  wrote integers; every score is stored unchanged as `float64`.
- **The challenge sets differ in their references.** GAMBITP has no reference. AGGREE, FECE,
  LIGATUR and QEbreak have a reference translation, but the source data does not record its type
  (`ref_type` is `unknown`) and the findings paper does not say how it was produced. The
  RefQuality sets have pseudo-references or post-edits.
- **Challenge sets are not scored like the official set.** Their authors analysed the results
  with their own code. Which of the two hypotheses of a challenge item is which is not stated in
  the findings paper for several sets; see `challenge_sets.hyp_semantics`.
- **`t1_instruction_fault` is null in every row**, and `para_idx` is null for FECE and QEbreak
  items.

## Coverage of the official test set

One line per official language pair: its source items, its pairs (candidate translations), the
type of its reference and whether it is a priority pair.

{{coverage_table}}

The counts are those of the `items` and `pairs` configs; the release checks them against Table 2
of the findings paper. `native` is the "Human" reference of that table. The challenge sets are not
in this table: they hold {{n_challenge_items}} items and {{n_challenge_pairs}} pairs.

## Field glossary

The same tables are in
[`docs/FORMAT.md`](https://github.com/wmt-conference/wmt26-mteval/blob/main/docs/FORMAT.md),
and the descriptions are stored in the schema of every Parquet file.

{{field_tables}}

## Known issues and errata

The findings paper and the data differ in a few places, such as the number of official pairs, the
number of translations per source, the deduplication rule of the baselines and the item counts of
two challenge sets. They are listed, with the figures on both sides, in
[`docs/ERRATA.md`](https://github.com/wmt-conference/wmt26-mteval/blob/main/docs/ERRATA.md).
The release process is in
[`docs/RELEASE.md`](https://github.com/wmt-conference/wmt26-mteval/blob/main/docs/RELEASE.md).

## Personal data

- The model cards of the participants ask for contact details and for consent. `systems` is built
  from a fixed allow-list of answers. The fields that hold the names and e-mail addresses of the
  submitters and the answers about data collection are not on the list, and their answers were
  never read. The allow-listed answers were screened too: an e-mail address inside an
  allow-listed answer is replaced by `[email removed]`, and one allow-listed answer that named
  individuals was replaced by the organisations they belong to. The only e-mail address in the
  metadata is the organisers' mailing list, below. A release check looks for any other address
  in `systems`, `challenge_sets`, this card and the build manifest.
- The links in `systems.link` were volunteered by the teams, in the model card, for publication.
- Source texts and translations are published as they are in the WMT26 General MT test set, and
  are not changed or scanned. They may contain personal details that occur in the original
  public texts, such as social-media posts.
- Annotator ids in the raw annotation records of `grand` are pseudonyms.

If you find personal data that should not be here, write to
`wmt-qe-metrics-organizers@googlegroups.com`.

## Human annotations

The human evaluation (contrastive error span annotation, cESA) was carried out by the WMT26
General MT task (Kocmi et al., 2026a; Zouhar et al., 2026a). Its data, including the raw
annotation logs, are public in General MT's release:

<https://github.com/wmt-conference/wmt26-general-mt/releases/tag/humeval>

v0.9 contains no human-annotation gold in its typed configs; a gold config is planned for v1.0.
The unchanged `grand` config still holds the organisers' raw annotation records in the column
`ground_truth__record`.

The speech items point to video files and some social-media items to screenshots, in
`items.video_path` and `items.screenshot_path`. These paths are relative to General MT's
multimodal assets, the archive `multimodal_inputs.zip` in the release linked above.

## Provenance

The typed configs are derived only from `wmt26-grand.parquet` (sha256 `{{grand_sha256}}`), the
scrubbed model cards, the list of priority pairs, and the curated tables of names, language codes
and challenge sets that follow the findings paper. `build_manifest.json` lists the sha256 and the
rows of every Parquet file, and the GitHub repository lists the source files in
`data/MANIFEST.tsv`. The code that builds and checks this repository is at
<https://github.com/wmt-conference/wmt26-mteval>.

## Citation

If you use this data, cite the findings paper:

```bibtex
@inproceedings{lavie2026findings,
  title = {Findings of the WMT26 Shared Task on Automated Translation Quality Evaluation Systems: Compact Open Models are Competitive Quality Evaluators},
  author = {Lavie, Alon and
            Hanneman, Greg and
            Perrella, Stefano and
            Ding, Shuoyang and
            Avramidis, Eleftherios and
            Proietti, Lorenzo and
            Lo, Chi-kiu and
            Shurtz, Ammon and
            Zerva, Chrysoula and
            Sindhujan, Archchana and
            Zouhar, Vilém and
            Kanojia, Diptesh and
            Blain, Frédéric and
            Thompson, Brian and
            Filandrianos, Giorgos and
            Menis Mastromichalakis, Orfeas and
            Kocmi, Tom and
            Gupta, Pranav},
  booktitle = {Proceedings of the Eleventh Conference on Machine Translation (WMT)},
  year = {2026},
  pages = {931--1030},
  url = {https://www2.statmt.org/wmt26/pdf/2026.wmt-1.49.pdf}
}
```

The data build on the following work, as the findings paper cites it.

The WMT26 General MT task, which provided the source texts, the translations and the human
evaluation:

```bibtex
@inproceedings{kocmi2026general,
  title = {Findings of the WMT26 general machine translation shared task: Contrastive dynamic human evaluation at scale},
  author = {Kocmi, Tom and Artemova, Ekaterina and Avramidis, Eleftherios and Bawden, Rachel and Bojar, Ondřej and Dukanov, Sergey and Dvorkovich, Anton and Fishel, Mark and Freitag, Markus and Frontull, Samuel and Gowda, Thamme and Grundkiewicz, Roman and Haddow, Barry and Kharevich, Stan and Koehn, Philipp and Li, Zheng and Maillard, Jean and Monz, Christof and Murauski, Alexander and others},
  booktitle = {To appear in Proceedings of the Eleventh Conference on Machine Translation},
  year = {2026},
  publisher = {Association for Computational Linguistics}
}
```

The contrastive error span annotation protocol (cESA) of the human evaluation:

```bibtex
@misc{zouhar2026cesa,
  title = {Contrastive esa: Human evaluation of multiple translations at once},
  author = {Zouhar, Vilém and Grundkiewicz, Roman and Rajaee, Sara and Riley, Parker and Popel, Martin and Bawden, Rachel and Koehn, Philipp and Carpuat, Marine and Kocmi, Tom},
  year = {2026},
  eprint = {2607.26640},
  archiveprefix = {arXiv}
}
```

The pipeline that built the pseudo-references:

```bibtex
@inproceedings{kanojia2026blind,
  title = {In the blind: Building pseudo-references for MT evaluation},
  author = {Kanojia, Diptesh and Lo, Chi-kiu and Sindhujan, Archchana and Larkin, Samuel and Hanneman, Greg and Lavie, Alon},
  booktitle = {Proceedings of the Eleventh Conference on Machine Translation},
  year = {2026},
  publisher = {Association for Computational Linguistics}
}
```

The challenge sets (`RefQuality-PD` and `RefQuality-PE` are the organisers' own, in Section 5.2.1
of the findings paper):

```bibtex
@inproceedings{griesbeck2026aggree,
  title = {AGGREE: An aggregation-grounded challenge set for segment-level MT metric evaluation},
  author = {Griesbeck, Lena and Geierhos, Michaela},
  booktitle = {Proceedings of the Eleventh Conference on Machine Translation},
  year = {2026},
  address = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics}
}

@inproceedings{ding2026fece,
  title = {FECE: A challenge set for evaluating fine-grained translation error correction},
  author = {Ding, Jie and Duan, Xiangyu},
  booktitle = {Proceedings of the Eleventh Conference on Machine Translation},
  year = {2026},
  address = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics}
}

@inproceedings{menismastromichalakis2026gambit,
  title = {Benchmarking gender bias in machine translation evaluation metrics across occupations},
  author = {Menis Mastromichalakis, Orfeas and Filandrianos, Giorgos and Mohammed, Wafaa and Attanasio, Giuseppe and Zerva, Chrysoula},
  booktitle = {Proceedings of the Eleventh Conference on Machine Translation},
  year = {2026},
  address = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics}
}

@inproceedings{mishra2026ligatur,
  title = {Translation metrics cannot judge what their tokeniser deletes: LIGATUR and AEGIS submissions to the WMT26 shared task on automated translation quality evaluation},
  author = {Mishra, Animesh and Sharma, Krishang and Khetarpaul, Sonia},
  booktitle = {Proceedings of the Eleventh Conference on Machine Translation},
  year = {2026},
  address = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics}
}

@inproceedings{silchenko2026qebreak,
  title = {QEbreak at WMT26: A preregistered audit of omission and addition asymmetry in machine translation quality metrics},
  author = {Silchenko, Maksim},
  booktitle = {Proceedings of the Eleventh Conference on Machine Translation},
  year = {2026},
  address = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics}
}
```

## Contact

The organisers: `wmt-qe-metrics-organizers@googlegroups.com`. Web page of the shared task:
<https://www2.statmt.org/wmt26/mteval-task.html>.
