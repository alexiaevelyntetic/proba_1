"""Step 6: which mistakes would cost most, and where the mapping is weakest.

Two different costs, kept apart on purpose:

* Mapping risk (our mistake): the category is mapped to the wrong place, so its core
  products (those the model places in the chosen category) all land on the wrong page.
      mapping_risk_eur = P(mapping wrong | confidence) x core revenue
  P(mapping wrong) is calibrated on grouped CV: held-out categories whose labels all agree
  give a clean answer to check the chosen id against.
* Stray revenue (not fixable by any mapping): products the model places somewhere other
  than the rest of their supplier category. One id per category misplaces them whatever we
  choose; they need product-level overrides.

Run with `python -m src.risk`.
"""
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.data import DATA_DIR, load
from src.evaluate import FOLDS, assign_folds
from src.map import build_mapping

OUT_DIR = DATA_DIR / "output"
CATCH_ALL = (r"^\s*(?:assorted|misc|miscellaneous|new (?:lines|products|in|listings)|promo|promotions"
             r"|feature space|gondola end|sundries|launches)\b")


def is_catch_all(names: pd.Series) -> pd.Series:
    return names.str.contains(CATCH_ALL, case=False, regex=True)


def cross_validate(products, labels, categories):
    """Grouped CV. Returns (held-out labelled products, held-out categories with clean truth)."""
    labelled = labels.merge(products[["supplier_sku", "key"]], on="supplier_sku")
    folds = assign_folds(labelled)
    prod_frames, cat_frames = [], []
    for f in range(FOLDS):
        test = labelled[folds == f]
        mapping, prods = build_mapping(products, labels[folds != f], categories, return_products=True)
        held = test.merge(prods[["supplier_sku", "our_category_id", "revenue", "supplier_category"]],
                          on="supplier_sku", suffixes=("_truth", ""))
        prod_frames.append(held.merge(mapping[["key", "confidence", "purity"]], on="key"))

        votes = test.groupby("key").our_category_id.agg(n="size", distinct="nunique", truth="first")
        clean = votes[(votes.n >= 2) & (votes.distinct == 1)]
        cat_frames.append(mapping.merge(clean, left_on="key", right_index=True))
    prods = pd.concat(prod_frames, ignore_index=True)
    prods["ok"] = prods.our_category_id_truth == prods.our_category_id
    cats = pd.concat(cat_frames, ignore_index=True)
    cats["ok"] = cats.truth == cats.our_category_id
    return prods, cats


def weak_spots(cv_prods, cv_cats, paths):
    """Held-out accuracy by segment, plus the most frequent confusions."""
    cv_prods = cv_prods.assign(catch_all=is_catch_all(cv_prods.supplier_category),
                               dept=cv_prods.our_category_id_truth.map(paths).str.split(" > ").str[0])
    acc = lambda g: pd.Series({"labels": len(g), "accuracy": g.ok.mean()})
    segments = {
        "category confidence": cv_prods.groupby(pd.cut(cv_prods.confidence, [0, .4, .6, .8, 1.0]),
                                                observed=True).apply(acc),
        "category purity": cv_prods.groupby(pd.cut(cv_prods.purity, [0, .5, .75, .99, 1.0]),
                                            observed=True).apply(acc),
        "catch-all name": cv_prods.groupby("catch_all").apply(acc),
        "our department (truth)": cv_prods.groupby("dept").apply(acc).sort_values("accuracy"),
    }
    wrong = cv_cats[~cv_cats.ok].assign(truth_path=lambda d: d.truth.map(paths),
                                        chosen_path=lambda d: d.our_category_id.map(paths))
    confusions = wrong.groupby(["truth_path", "chosen_path"]).size().sort_values(ascending=False)
    return segments, confusions


def main():
    products, labels, _, categories = load()
    paths = dict(zip(categories.our_category_id, categories.path))

    cv_prods, cv_cats = cross_validate(products, labels, categories)
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(cv_cats.confidence, cv_cats.ok)
    calib = cv_cats.groupby(pd.cut(cv_cats.confidence, [0, .4, .5, .6, .7, .8, .9, 1.0]), observed=True).agg(
        categories=("ok", "size"), mapped_right=("ok", "mean"))

    mapping, prods = build_mapping(products, labels, categories, return_products=True)
    prods["core_revenue"] = prods.revenue.where(prods.pred_id == prods.our_category_id, 0)
    per_key = prods.groupby("key").agg(core_revenue=("core_revenue", "sum"))
    risk = mapping.merge(per_key, left_on="key", right_index=True)
    risk["supplier_category"] = risk.key.map(products.groupby("key").supplier_category.first())
    risk["stray_revenue"] = risk.revenue - risk.core_revenue
    risk["p_mapping_wrong"] = 1 - iso.predict(risk.confidence)
    risk["mapping_risk_eur"] = risk.p_mapping_wrong * risk.core_revenue
    risk["our_path"] = risk.our_category_id.map(paths)
    risk["runner_up_path"] = risk.runner_up_id.map(paths)
    risk["labels_disagree"] = risk.label_majority_id.notna() & (risk.label_majority_id != risk.our_category_id)
    risk["catch_all_name"] = is_catch_all(risk.supplier_category)
    risk = risk.sort_values("mapping_risk_eur", ascending=False)
    risk["cum_share_of_risk"] = risk.mapping_risk_eur.cumsum() / risk.mapping_risk_eur.sum()

    cols = ["supplier_category", "our_category_id", "our_path", "runner_up_path", "revenue", "core_revenue",
            "p_mapping_wrong", "mapping_risk_eur", "cum_share_of_risk", "stray_revenue", "confidence", "purity",
            "n_products", "n_labels", "labels_disagree", "catch_all_name"]
    OUT_DIR.mkdir(exist_ok=True)
    risk[cols].to_csv(OUT_DIR / "risk_ranking.csv", index=False, float_format="%.4f")

    # Where the model disagrees with the category: candidate product-level overrides.
    strays = prods[prods.pred_id != prods.our_category_id].assign(
        mapped_to=lambda d: d.our_category_id.map(paths), model_suggests=lambda d: d.pred_id.map(paths))
    strays.sort_values("revenue", ascending=False)[
        ["supplier_sku", "supplier_category", "product_name", "revenue", "mapped_to", "model_suggests"]
    ].to_csv(OUT_DIR / "stray_products.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 50)
    total = products.revenue.sum()
    print("== Category-level calibration (held-out categories with unanimous labels)")
    print(calib.to_string(float_format="{:.3f}".format))
    print(f"overall: {cv_cats.ok.mean():.1%} of {len(cv_cats)} held-out categories mapped right")

    print(f"\n== Mapping risk: EUR {risk.mapping_risk_eur.sum():,.0f} expected misplaced "
          f"({risk.mapping_risk_eur.sum() / total:.1%} of EUR {total:,.0f})")
    for n in (20, 100, 300):
        print(f"  top {n} categories hold {risk.cum_share_of_risk.iloc[n - 1]:.0%} of it")
    print("\nTop 20 by mapping risk:")
    print(risk[["supplier_category", "our_path", "runner_up_path", "core_revenue", "p_mapping_wrong",
                "mapping_risk_eur", "n_labels"]].head(20).to_string(index=False, float_format="{:,.2f}".format))

    print(f"\n== Stray revenue (unfixable by one id per category): EUR {risk.stray_revenue.sum():,.0f} "
          f"({risk.stray_revenue.sum() / total:.1%}), {len(strays)} products")
    print(risk.nlargest(10, "stray_revenue")[["supplier_category", "our_path", "revenue", "stray_revenue"]]
          .to_string(index=False, float_format="{:,.0f}".format))

    segments, confusions = weak_spots(cv_prods, cv_cats, paths)
    for name, table in segments.items():
        print(f"\n== Held-out product accuracy by {name}")
        print(table.to_string(float_format="{:.3f}".format))
    print("\n== Category-level confusions on held-out categories (truth -> chosen)")
    print(confusions.head(15).to_string())


if __name__ == "__main__":
    main()
