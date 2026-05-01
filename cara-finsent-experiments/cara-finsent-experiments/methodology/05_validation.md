# 05 · Validation

The dataset is gated by a small but opinionated `pytest` suite that has to
pass before any experiment is considered legitimate.

* File: [`tests/test_dataset_quality.py`](../tests/test_dataset_quality.py)
* Run: `pytest tests/test_dataset_quality.py -v`
* Fixture: a single module-scoped `pd.read_csv` of
  [`data/processed/latest/dataset.csv`](../data/processed/latest/dataset.csv).
  If the canonical dataset doesn't exist yet the suite **skips** rather
  than fails, so a fresh clone never blows up before the first build.

## The 14 checks (and why they exist)

| # | Test | Catches |
|---:|---|---|
| 1 | `test_dataset_exists` | empty / zero-row build |
| 2 | `test_required_columns_present` | drift in writer schema |
| 3 | `test_label_values_normalised` | leftover `bullish/bearish/0/1/2` strings |
| 4 | `test_no_empty_text_clean` | preprocessing produced empty rows that survived |
| 5 | `test_token_length_in_range` | `n_tokens ∉ [3, 512]` — i.e. filter regression |
| 6 | `test_no_url_or_at_user_or_cashtag_in_text_clean` | masking regression on a 2k random sample |
| 7 | `test_test_set_is_substantial` | test < 500 rows = unreliable metrics |
| 8 | `test_test_set_has_all_three_labels` | tri-class problem with a missing class is meaningless |
| 9 | `test_test_set_minority_class_size` | minority class < 50 — F1 too noisy |
| 10 | `test_train_imbalance_within_threshold` | balancer regression (≤ 2.5×) |
| 11 | `test_no_train_test_text_overlap` | exact leakage from train into test |
| 12 | `test_tier_values_valid` | a new source slipped in with an unknown tier |
| 13 | `test_at_least_one_gold_test_per_source` | weak proxy for source diversity in test |
| 14 | `test_within_source_no_exact_dupes` | step-3 dedup regression for a specific source |

## Pass log for build `20260430_212903`

```
collected 14 items
tests/test_dataset_quality.py::test_dataset_exists                  PASSED
tests/test_dataset_quality.py::test_required_columns_present        PASSED
tests/test_dataset_quality.py::test_label_values_normalised         PASSED
tests/test_dataset_quality.py::test_no_empty_text_clean             PASSED
tests/test_dataset_quality.py::test_token_length_in_range           PASSED
tests/test_dataset_quality.py::test_no_url_or_at_user_or_cashtag…   PASSED
tests/test_dataset_quality.py::test_test_set_is_substantial         PASSED
tests/test_dataset_quality.py::test_test_set_has_all_three_labels   PASSED
tests/test_dataset_quality.py::test_test_set_minority_class_size    PASSED
tests/test_dataset_quality.py::test_train_imbalance_within_threshold PASSED
tests/test_dataset_quality.py::test_no_train_test_text_overlap      PASSED
tests/test_dataset_quality.py::test_tier_values_valid               PASSED
tests/test_dataset_quality.py::test_at_least_one_gold_test_per_source PASSED
tests/test_dataset_quality.py::test_within_source_no_exact_dupes    PASSED

============================= 14 passed in 1.13s ==============================
```

## Adding a new check

Drop a function `def test_<name>(df): …` into the same file. The `df`
fixture handles loading. Keep checks fast — they run on every PR.
