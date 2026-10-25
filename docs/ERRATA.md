# Errata

Where the findings paper (Lavie et al., 2026, WMT, pp. 931-1030) and the v0.9 data differ.

### 1. Official segment pairs
Section 3.1 says 252,919; Table 2 and the data say 252,923 (plus 110,319 challenge pairs = 363,242).

### 2. Translations per source segment
Section 3.1 says 23 to 47; each official item has 27 to 31 (47 distinct candidates overall).

### 3. "18 direct Task 3 submissions"
Table 10 has 18 rows: 9 direct submissions and 9 baselines. The data hold 13 direct runs.

### 4. Baseline deduplication
Sections 4.2 and 6.3.1 describe different rules; the organisers' files follow Section 4.2.
`baselines.variant = 'official'` is the deduplicated snapshot (provisional, to be checked against
the paper in v1.0); `alt` is the original, with Task 1 columns only.

### 5. Challenge-set item counts
Table 15 gives 1,749 QEbreak and 159 LIGATUR items; Table 14 and the data give 2,886 and 171.

### 6. Baseline names
Names differ across Tables 4, 8 and 10; `systems.canonical_name` gives one name per run.

### 7. GPT-5.6 Sol xHigh baseline
The paper says it completed 6 of the 8 pairs. Its `t2_score` is a placeholder (50.0) and its
`eng-sme` and `eng-ind` Task 1 predictions are mostly empty: do not use them.
