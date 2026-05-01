# 01 · Dataset collection

CARA-FinSent fuses **nine** publicly available financial-sentiment datasets
into one tri-class corpus. Every source is collected by a small,
self-contained script that writes a timestamped CSV under
`data/raw/<source>/<source>_YYYYMMDD_HHMMSS.csv`. Re-running a collector
never overwrites old snapshots — it adds a new file, and the build script
always picks the latest.

## Inventory

| Source | Tier | Rows in build* | License | Collector | HF repo |
|---|---|---:|---|---|---|
| financial_phrasebank | gold | 4,735 | CC BY-NC-SA 3.0 | [scripts/00_prepare_phrasebank_fiqa.py](../scripts/00_prepare_phrasebank_fiqa.py) | `takala/financial_phrasebank` |
| fiqa | gold | 1,100 | CC BY-NC-SA 4.0 | [scripts/00_prepare_phrasebank_fiqa.py](../scripts/00_prepare_phrasebank_fiqa.py) | `LiyuanLucasLiu/fiqa-2018` (task1) |
| auditor_sentiment | gold | 50 | unknown — research use | [scripts/04_collect_extra_datasets.py](../scripts/04_collect_extra_datasets.py) | `demo-org/auditor_sentiment_review` |
| fomc_sentiment | gold | 1,970 | CC BY 4.0 | [scripts/09_collect_fomc_sentiment.py](../scripts/09_collect_fomc_sentiment.py) | `gtfintechlab/fomc_communication` |
| semeval2017_task5 | gold | 5,088 | CC BY 4.0 | [scripts/08_collect_semeval2017.py](../scripts/08_collect_semeval2017.py) | `TimKoornstra/financial-tweets-sentiment` (with `sismetanin/semeval2017_task5_*` fall-backs) |
| financial_classification | silver | 96 | unknown | [scripts/04_collect_extra_datasets.py](../scripts/04_collect_extra_datasets.py) | `nickmuchi/financial-classification` |
| fingpt_sentiment | silver | 560 | research-use only | [scripts/04_collect_extra_datasets.py](../scripts/04_collect_extra_datasets.py) | `FinGPT/fingpt-sentiment-train` |
| twitter_fin_news | silver | 1 | unknown | [scripts/04_collect_extra_datasets.py](../scripts/04_collect_extra_datasets.py) | `zeroshot/twitter-financial-news-sentiment` |
| twitter_fin_topic | silver | 316 | unknown | [scripts/04_collect_extra_datasets.py](../scripts/04_collect_extra_datasets.py) | `zeroshot/twitter-financial-news-topic` |

\* Counts are for build `20260430_212903`. Numbers shrink between raw and
final because of dedup (Section 03), language filtering and class balancing.

> The unknown licenses are inherited from the upstream Hugging Face cards.
> They are flagged in [`data_card.md`](../data/processed/latest/data_card.md)
> so reviewers can decide whether to ship them.

## Tier definition

```
gold      → human-annotated, peer-reviewed, well-documented
silver    → community / weakly-supervised, plausible labels
synthetic → generated (paraphrasing, LLM rewrites). Currently zero.
bronze    → noisy / scraped / non-financial spillover. Currently zero.
```

The tier is assigned **at ingest time** in
[scripts/05_build_dataset.py](../scripts/05_build_dataset.py) by a fixed
mapping (`GOLD_SOURCES`, `SILVER_SOURCES`, `BRONZE_SOURCES`). Tier drives:

* **dedup precedence** (gold survives a near-duplicate collision)
* **test-set carving** (test rows are gold-only)
* **per-row sample weight** at training time (`gold=1.0, silver=0.5, …`)

## Reproducing the raw layer

```bash
# Hugging Face token must be exported first; collectors rely on `datasets`
$env:HF_TOKEN = "<your_token>"            # PowerShell
python scripts/00_prepare_phrasebank_fiqa.py
python scripts/04_collect_extra_datasets.py
python scripts/08_collect_semeval2017.py
python scripts/09_collect_fomc_sentiment.py
```

After all four finish, `data/raw/` will contain ~9 sub-folders, each holding
at least one CSV with the columns `text, label, source, license` plus
optional metadata (`agreement`, `topic`, `meeting_date` etc.).

## Collector script contract

Every collector must:

1. Write to `data/raw/<source>/<source>_<UTC_timestamp>.csv`.
2. Emit at minimum the columns `text, label, source`.
3. Map labels to the `{negative, neutral, positive}` standard set
   (or leave them raw and let `normalize_label()` do it later — but in
   practice we normalise eagerly).
4. Print a final line of the form
   `[DONE] Saved <n> rows -> <path>` and a label histogram so the build
   script's smoke checks pass.

If a collector reaches a deprecated/removed dataset, it must `try/except`
and continue with the next candidate (see how
[scripts/08_collect_semeval2017.py](../scripts/08_collect_semeval2017.py)
falls through `TimKoornstra/financial-tweets-sentiment` →
`sismetanin/semeval2017_task5_subtask_a` → `…_subtask_b`).

## Known data-collection caveats

* **SemEval-2017 task 5 has no negative-only repo on the Hub anymore.** The
  TimKoornstra mirror returns positives + neutrals only, so SemEval cannot
  contribute negatives to training. Negatives mostly come from
  `financial_phrasebank` and `fomc_sentiment` (hawkish=negative).
* **FOMC sentiment** has only the `default` HF config — the script tries
  `meeting_minutes` / `press_conference` / `speeches` and falls through
  gracefully.
* **Auditor sentiment** repo is tiny (≤ 100 rows after filter) — kept for
  domain coverage, but does not move metrics.
* **Twitter financial-news** sentiment is mostly noisy weak labels; we
  keep it at silver tier and never put it in the test set.
