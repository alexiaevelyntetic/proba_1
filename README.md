# Supplier category mapping

This maps each of the supplier's ~3,000 category names to one of our 150 categories. It also
ranks which mistakes would cost the most and shows where the mapping is weakest.

**Deliverable:** [`output/mapping.csv`](output/mapping.csv) (`supplier_category,our_category_id`,
3,030 lines, one per distinct value as written).
**Cost and weak spots:** [`REPORT.md`](REPORT.md), [`output/risk_ranking.csv`](output/risk_ranking.csv),
[`output/stray_products.csv`](output/stray_products.csv).

## Running it

You need Python 3.10+. The input CSVs are not in the repo. Copy `our_categories.csv`,
`supplier_products.csv`, `labelled_products.csv` and `sales.csv` into the repo root first.

```bash
pip install -r requirements.txt
python -m src.map        # writes output/mapping.csv and mapping_detail.csv   (~10 s)
python -m src.risk       # writes risk_ranking.csv, stray_products.csv         (~1 min)
python -m unittest -v    # 11 tests, including format checks on the real output
```

Optional:

- `python -m src.explore` prints the data checks.
- `python -m src.baseline` runs the labels-only baseline.
- `python -m src.evaluate` runs the grouped-CV parameter search (~3 min).

| File | What it does |
|---|---|
| `src/data.py` | Loads the inputs. `category_key` folds case and whitespace. |
| `src/model.py` | Product classifier and per-category aggregation. |
| `src/map.py` | Builds the final mapping. |
| `src/evaluate.py` | Grouped cross-validation. |
| `src/risk.py` | Calibrated cost of mistakes, stray products, weak segments. |

## What I did, how I checked it, what I left out

**What I did.** Supplier names that differ only in case or spacing are folded into one key,
which leaves 2,678 real categories. Every product is classified on its own: TF-IDF character
n-grams of the name with the brand removed, plus pack unit and size band, fed into logistic
regression trained on the 4,018 hand-labelled products. Brands carry no signal: a brand's
most common category holds only ~40% of its labelled products. The size band matters because our tree splits catering from
retail packs at 5 kg / 5 l. Each supplier category gets the category with the highest summed
probability over *all* its products. Hand labels are added as small extra votes.

**How I checked it.** I used grouped cross-validation: whole supplier categories are held out,
so they look exactly like the 1,051 categories that have no labels.

| Method | Held-out labelled products placed correctly |
|---|---|
| Baseline: majority of hand labels | 80.8% |
| This model on categories with **no** labels at all | **84.8%** |
| Ceiling for any one-id-per-category answer | ~88% |
| Held-out categories with unanimous labels mapped right | 96.9% |

The 88% ceiling exists because supplier categories are mixed. Unit tests check the output
contract: one line per raw value, valid ids only, and identical answers for case or space
variants. They also cover the vote tie-breaks and pack-size parsing. I read through the
mapping changes that each modelling decision caused (for example, all 37 lines the pack-size
feature moved) and the top of the risk list.

**What I deliberately did not do.**

- I did not split a supplier category across several ids, because the brief requires one.
  Instead, `stray_products.csv` lists ~10k products that this rule will misplace.
- I did not weight the mapping by revenue. In CV it scored the same, but it let a few big
  sellers overrule the bulk of a category. Revenue is used only to rank risk.
- I did not use the supplier's category name as a model feature (it did not help in CV), and I
  did not hand-correct individual answers.
- I did not use an LLM or embeddings. A linear model is reproducible, runs in seconds and was
  enough here.
- 6 of our categories never appear in the labels, so the model cannot predict them.

## Evidence

Before and after, on held-out supplier categories:

| | labels-only baseline | model, no labels | model + pack size |
|---|---|---|---|
| products placed correctly | 80.8% | 84.1% | 84.8% |
| catering/retail products correct | – | 61% | 92% |
| expected misplaced revenue (mapping errors) | – | EUR 12.9M | EUR 8.9M |

Output sample (`output/mapping_detail.csv`; spelling variants share one answer):

```
supplier_category      our_category_id  confidence  path
APLS PRS               7245             0.62        Fruit & veg > Fruit > Apples & pears
APLSPRS                7245             0.62        Fruit & veg > Fruit > Apples & pears
ALE STOUT              6845             0.58        Drinks > Beer & cider > Ale & stout
FL > TRADE             7623             0.69        Food cupboard > Baking > Flour, catering packs (5 kg and over)
```

## AI use

<!-- TODO(you): check that this line matches what you actually did, and edit it. -->
I used Claude (an AI coding assistant) to write the code and run the analysis. By hand, I
checked the mapping changes, the top of the risk ranking and sample categories against
their products.

## Found outside the task, not fixed

The details are in [`REPORT.md`](REPORT.md#found-outside-the-task-not-fixed).

- Many supplier categories mix unrelated products, and big sellers sit in the wrong category.
  Moving products category by category misplaces up to ~31% of revenue, whatever the mapping.
- Some hand labels look like strays themselves.
- `pack_size` comes in ~15 formats.
- 2,398 products have zero sales.
- 287 categories are spelled several ways.
