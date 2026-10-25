---
pretty_name: WMT26 MT-Eval
license: other
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
  - split: test
    path: data/items/test-*.parquet
- config_name: pairs
  default: true
  data_files:
  - split: test
    path: data/pairs/test-*.parquet
- config_name: submissions
  data_files:
  - split: test
    path: data/submissions/test-*.parquet
- config_name: baselines
  data_files:
  - split: test
    path: data/baselines/test-*.parquet
- config_name: systems
  data_files:
  - split: test
    path: data/systems/test-*.parquet
- config_name: mt_systems
  data_files:
  - split: test
    path: data/mt_systems/test-*.parquet
- config_name: challenge_sets
  data_files:
  - split: test
    path: data/challenge_sets/test-*.parquet
- config_name: grand
  data_files:
  - split: test
    path: wmt26-grand.parquet
---

# WMT26 MT-Eval

Test data and evaluator predictions of the **WMT26 Shared Task on Automated Translation Quality
Evaluation Systems**, the shared task on MT metrics and quality estimation (findings paper: Lavie
et al., 2026; see [Citation](#citation)).

Participants built automatic evaluators of translation quality. The organisers gave them one test
set, made of source texts and candidate translations taken from the WMT26 General Machine
Translation task (General MT), and asked for predictions of four kinds:

| task | what it asks for | columns |
| --- | --- | --- |
| Task 1 | error spans with a severity, and omissions | `t1_errors`, `t1_omission` |
| Task 2 | a quality score for each translation, from 0 to 100 | `t2_score` |
| Task 3 | whether a translation is free of errors | `t3_error_free` |
| Task 4 | challenge sets, submitted by research teams, that probe the evaluators | `challenge_sets`, `challenge_set` |

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
- The licence is to be confirmed; see [Licence](#licence).

## Quick start

Every typed config has one split, named `test`, and loads with `datasets>=2.18`:

```python
from datasets import load_dataset

pairs = load_dataset("wmt/wmt26-mteval-data", "pairs", split="test")
print(pairs[0]["lp"], pairs[0]["src"], pairs[0]["hyp"], sep="\n")
```

`pairs` is the default config. `submissions` and `baselines` are large ({{n_submission_rows}}
and {{n_baseline_rows}} rows), so you may prefer to stream them or to download one config:

```python
subs = load_dataset("wmt/wmt26-mteval-data", "submissions", split="test", streaming=True)
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
grand = load_dataset("wmt/wmt26-mteval-data", "grand", split="test")
```

## Configs

{{config_table}}

- Join the hypotheses to the predictions on `(item_id, system)`, and the predictions to
  `systems` on `run_id`.
- `baselines.variant` is `official` for the organisers' released predictions (the deduplicated
  LLM baselines; the xCOMET files are identical before and after deduplication) and `alt` for the
  original, not deduplicated LLM outputs, which keep their Task 1 columns only. The baseline runs
  that predate the deduplication fix are not in the typed configs.
- A run covers only some items, and has no rows for the others; `systems.n_items`, `lps` and
  `covers_all_priority` say what it covers.
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

- **References are among the candidate translations.** Every available reference is also a
  hypothesis (`system_type` is `human_scratch`, `human_postedit` or `pseudo_ref`).
  {{n_reference_candidates}} pairs have a hypothesis that equals the item's reference, up to
  surrounding whitespace, and carry `is_reference_candidate = true`. A reference-based evaluator
  is given the reference as well, so it sees a copy of it on those rows: exclude them, or analyse
  them apart, if that matters to you.
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
- **Challenge sets are not scored like the official set.** They have no references here, and
  their authors analysed the results with their own code. Which of the two hypotheses of a
  challenge item is which is not stated in the findings paper for several sets; see
  `challenge_sets.hyp_semantics`.
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

- The model cards of the participants ask for contact details and for consent. The names and
  e-mail addresses of the submitters and their answers about data collection were never read
  into this release: `systems` is built from a fixed allow-list of answers. The only e-mail
  address in the metadata is the organisers' mailing list, below. A release check looks for any
  other address in `systems`, `challenge_sets`, this card and the build manifest.
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
rows of every file, and the GitHub repository lists the source files in `data/MANIFEST.tsv`. The
code that builds and checks this repository is at
<https://github.com/wmt-conference/wmt26-mteval>.

## Licence

The licence of this release is to be confirmed. Until it is, the `license` field of this card is
`other` and no licence is granted by this card. The data come from the WMT26 General MT task, so
read the terms of its release as well.

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
