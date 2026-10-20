# Cost of mistakes and weak spots

Numbers come from `python -m src.risk`, which writes `output/risk_ranking.csv` (one row per
supplier category, sorted by risk) and `output/stray_products.csv`. Total supplier revenue
is EUR 357.9M across 2,678 supplier categories (after folding case and spacing).

## Which mistakes would cost most

There are two different costs here, and they need different fixes.

**1. Mapping a category to the wrong place (our mistake).** If a category is mapped wrong,
its core products (the ones the model would put in the chosen category) all land on the
wrong page.

    mapping_risk_eur = P(mapping wrong) x core revenue

`P(mapping wrong)` comes from the category's confidence, calibrated on grouped
cross-validation. In that CV, whole supplier categories are held out, and 712 of them have
unanimous labels, which gives a clean answer to check against:

| confidence | held-out categories | mapped right |
|---|---|---|
| < 0.5 | 49 | 63% |
| 0.5 - 0.6 | 42 | 93% |
| >= 0.6 | 621 | ~100% |

Overall, 96.9% of held-out categories are mapped right. The expected misplaced revenue from
mapping errors is about **EUR 8.9M (2.5%)**. The top 20 categories hold 18% of it and the
top 100 hold 49%. The 278 categories with P(wrong) > 0.2 cover 14% of revenue but **70% of
the risk**, so they are the review queue.

Highest risk (see the CSV for the full list):

| supplier category | mapped to | runner-up | core EUR | P(wrong) |
|---|---|---|---|---|
| Ready > Ch.ready.me & Prprd Slds | Prepared salads | Chilled ready meals | 366k | 0.38 |
| Citr.frt | Citrus fruit | Berries | 397k | 0.32 |
| Pet ca and Cat food | Pet care | Cat food | 388k | 0.32 |
| PLANT-BASED.DRNKS | Plant-based drinks | Fresh milk | 371k | 0.32 |
| bee/ale.st | **Crisps** | Ale & stout | 345k | 0.32 |
| crsps > crckr | Crackers | Other juices | 249k | 0.38 |
| dog-fd | Dog food | Nappies & wipes | 242k | 0.32 |
| Shwr Bat | **Eggs** | Shower & bath | 204k | 0.32 |

Two of these are likely errors, and in both the model goes against the category name.
`bee/ale.st` holds 27 ales and 25 crisps, but the crisps are classified with more confidence
and win the probability sum. `Shwr Bat` holds more eggs than shower products. The model
never sees the category name, because using it scored no better in CV.

**2. Stray products (no mapping can fix them).** The model places about **10,100 products
(EUR 110M, 31% of revenue)** somewhere other than where their supplier category goes. This
is an upper bound: the product classifier is about 95.8% accurate, so part of it is model
error. The rest is real. Many supplier categories are two of our categories joined together
(`DOG-FOOD + NAPPIES WIPE`, `Citr.frt` with berries and soup, `ice cream` with lollies and
ready meals). Some single strays are big sellers. *Ostvik Lollies* in `ice cream` alone is
EUR 386k, and *Simply Pilchards* in `ME > PORK` is EUR 256k. Moving these by category puts
them on the wrong page whatever id we pick. `stray_products.csv` lists them by revenue as
candidates for product-level overrides.

## Where the mapping is weakest

Accuracy here is per labelled product, on held-out categories.

- **Catch-all supplier categories** (`Promo ...`, `New lines ...`, `Misc ...`,
  `Feature space ...`, `Gondola end ...`; 70 categories, 2.9% of revenue): **30%** accuracy,
  against 87% elsewhere. They are random assortments, and one id is wrong for most products
  whatever we choose.
- **Mixed categories:** when fewer than half the products agree on a category, accuracy is
  39%. 954 categories (50% of revenue) have purity below 0.75.
- **Low confidence:** confidence below 0.4 gives 42% accuracy, while above 0.8 it is 98.5%.
  The `confidence` column is a usable review signal.
- **When the category name and its content disagree** (see `bee/ale.st` and `Shwr Bat`
  above), the content wins. If your answer key follows the names, these are errors.
- **Near-ties between sibling categories:** ice cream vs ice lollies, fruit vs natural
  yogurt, ale vs lager vs cider, still vs sparkling water.
- **Categories with no labels** are 1,051 of 2,678 and carry 57% of the risk.
- **Our categories nobody labelled** (6 of 150) can never be predicted.
- **Fruit & veg and Frozen** are the weakest departments (82-83%).

**Fixed: catering vs retail pack sizes.** This used to be the most common confusion (sugar,
flour, oil, rice, pasta). Our tree splits these at 5 kg / 5 l. The model now gets the pack
amount as a size band, and that threshold separates the labelled products perfectly. On
held-out categories:

- product accuracy on these categories went from 61% to 92%;
- category-level catering/retail confusions dropped from 10 to 1;
- overall product accuracy went from 94.5% to 95.8%;
- expected mapping risk dropped from EUR 12.9M to 8.9M.

37 output lines changed, for example `FL > TRADE`, `Flour - Bulk` and `OIL` now map to
catering packs.

## Found outside the task, not fixed

- Supplier categories mix unrelated products, and some big sellers sit in the wrong supplier
  category. Moving products category by category misplaces about 31% of revenue at most.
  This is a process problem rather than a mapping one.
- `pack_size` comes in about 15 formats (`0,1 kg`, `2x300 gr`, `0.2Kg`, `x12`, `single`).
- 2,398 products have zero revenue (and zero units).
- 287 categories are spelled several ways in the same file.
- 131 labelled categories get a different answer from the model than from their label
  majority. With only 2-3 labels per category, some hand labels look like strays. For
  example, the single label in `crsps > crckr` is a pineapple juice.
