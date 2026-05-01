# 07 · FinBERT fine-tuning

Script: [`scripts/11_run_finbert_baseline.py`](../scripts/11_run_finbert_baseline.py).
Backbone: `ProsusAI/finbert` (BERT-base, financial-news pretrained, 3-class
head reused).

## Model

```python
AutoTokenizer.from_pretrained('ProsusAI/finbert')
AutoModelForSequenceClassification.from_pretrained(
    'ProsusAI/finbert',
    num_labels=3,
    label2id={'negative': 0, 'neutral': 1, 'positive': 2},
    id2label={0: 'negative', 1: 'neutral', 2: 'positive'},
    ignore_mismatched_sizes=True,
)
```

The pretrained head already has `negative/neutral/positive` slots so we
keep them and just continue fine-tuning. `ignore_mismatched_sizes=True`
is defensive for cases where someone swaps in a different backbone.

## Training arguments

```python
TrainingArguments(
    output_dir            = f'models/finbert_{ts}',
    num_train_epochs      = 3,           # --epochs
    per_device_train_batch_size = 8,     # --batch_size
    per_device_eval_batch_size  = 8,
    learning_rate         = 2e-5,        # --learning_rate
    weight_decay          = 0.01,
    warmup_ratio          = 0.1,
    eval_strategy         = 'epoch',
    save_strategy         = 'epoch',
    load_best_model_at_end= True,
    metric_for_best_model = 'macro_f1',
    greater_is_better     = True,
    seed                  = 42,
    fp16                  = False,       # CPU build — no mixed precision
    report_to             = 'none',
)
EarlyStoppingCallback(early_stopping_patience=1)
```

Tokeniser: `max_length=128`, `truncation=True`, `padding='longest'`. We
truncate at 128 because (a) > 95 % of `text_clean` rows are ≤ 128 BERT
tokens and (b) anything longer makes CPU training painfully slow.

## Inputs

* Text column: `text_clean` (the masked, demojised one — *not* the
  lower-cased one; BERT cases matter).
* Label column: `label` (already in the standard set).
* Train/test split: from `df['split']`.

## Compute footprint

On the reference machine (Intel Core 7 150U, no GPU):

| Item | Value |
|---|---|
| Train rows | 12,711 |
| Steps / epoch | ≈ 1,589 |
| Wall-clock / epoch | ~ 2.5–3 h |
| Total (3 epochs + eval) | 8–10 h |
| Peak RAM | ~ 4 GB |
| Final model size | ~ 440 MB |

GPU runs are roughly 30–50× faster — see `requirements-gpu.txt`.

## Outputs

```
models/finbert_<ts>/                 # HF checkpoint dir, best epoch only
results/<YYYY-MM-DD>/finbert_<ts>.csv  # one-row metrics file
results/<YYYY-MM-DD>/finbert_<ts>_perclass.csv
results/<YYYY-MM-DD>/finbert_<ts>_confusion.json
```

Metrics columns match the classical baselines (so script 17 can compare
them apples-to-apples).

## Re-running notes

* Do **not** start a second FinBERT job concurrently with a first on this
  machine — they will both thrash the CPU and finish slower than serial.
* If you abort mid-train and re-run, delete the old `models/finbert_<ts>/`
  to free disk; HF will refuse to load a half-written checkpoint.
* The script fully respects `auto_detect_data()`, so building a fresh
  canonical dataset and re-running this script uses the new data
  automatically.
