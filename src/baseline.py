"""Step 3: map each supplier category by majority vote of its hand-labelled products.

Ties break on the labelled products' revenue, then on the lower category id, so the result
is deterministic. Categories with no labels are left unmapped here; a product-level model
fills them later. Run with `python -m src.baseline`.
"""
from pathlib import Path

import pandas as pd

from src.data import DATA_DIR, load

OUT_DIR = DATA_DIR / "output"


def vote(labelled: pd.DataFrame) -> pd.DataFrame:
    """labelled: rows of (key, our_category_id, revenue). Returns one row per key."""
    tally = (labelled.groupby(["key", "our_category_id"])
             .agg(votes=("our_category_id", "size"), revenue=("revenue", "sum"))
             .reset_index()
             .sort_values(["key", "votes", "revenue", "our_category_id"],
                          ascending=[True, False, False, True]))
    total = tally.groupby("key")["votes"].transform("sum")
    tally["share"] = tally["votes"] / total
    tally["n_labels"] = total
    return tally.drop_duplicates("key")[["key", "our_category_id", "share", "n_labels"]]


def leave_one_out(labelled: pd.DataFrame) -> pd.DataFrame:
    """For every labelled product in a category with >=2 labels, predict it from the others.

    This measures what we care about: how often a product moved by the category mapping
    lands where our team would put it.
    """
    rows = []
    for key, group in labelled.groupby("key"):
        if len(group) < 2:
            continue
        for idx in group.index:
            rest = group.drop(idx)
            pred = vote(rest).iloc[0]
            rows.append({"key": key, "supplier_sku": group.at[idx, "supplier_sku"],
                         "truth": group.at[idx, "our_category_id"], "pred": pred.our_category_id,
                         "revenue": group.at[idx, "revenue"], "n_other": len(rest)})
    return pd.DataFrame(rows)


def main():
    products, labels, _, categories = load()
    labelled = labels.merge(products[["supplier_sku", "key", "revenue"]], on="supplier_sku")
    mapping = vote(labelled)

    # One output line per raw spelling; folded spellings share the answer by construction.
    raw = products[["supplier_category", "key"]].drop_duplicates()
    out = raw.merge(mapping, on="key", how="left")
    revenue = products.groupby("key")["revenue"].sum().rename("category_revenue")
    out = out.merge(revenue, on="key").sort_values("supplier_category")

    OUT_DIR.mkdir(exist_ok=True)
    out.to_csv(OUT_DIR / "baseline_mapping_detail.csv", index=False)
    print(f"mapped {out.our_category_id.notna().sum()} / {len(out)} raw values "
          f"({mapping.shape[0]} / {products.key.nunique()} folded categories)")
    print(f"revenue covered: {out.drop_duplicates('key').query('our_category_id.notna()').category_revenue.sum() / products.revenue.sum():.1%}")

    loo = leave_one_out(labelled)
    loo["ok"] = loo.truth == loo.pred
    print(f"\nleave-one-out on {len(loo)} labelled products in {loo.key.nunique()} categories:")
    print(f"  accuracy {loo.ok.mean():.1%}, revenue-weighted {(loo.ok * loo.revenue).sum() / loo.revenue.sum():.1%}")
    by_n = loo.assign(n=loo.n_other.clip(upper=4)).groupby("n").ok.agg(["mean", "size"])
    by_n.index = [f"{n}{'+' if n == 4 else ''} other labels" for n in by_n.index]
    print(by_n.rename(columns={"mean": "accuracy", "size": "products"}).to_string(float_format="{:.1%}".format))

    paths = dict(zip(categories.our_category_id, categories.path))
    names = products.groupby("key")["supplier_category"].first()
    weak = mapping[mapping.share < 1].merge(revenue, on="key").nlargest(10, "category_revenue")
    print("\nHighest-revenue categories whose labels disagree:")
    for r in weak.itertuples():
        print(f"  {names[r.key]!r:40} -> {paths[r.our_category_id]:45} share {r.share:.0%} of {r.n_labels}, "
              f"EUR {r.category_revenue:,.0f}")


if __name__ == "__main__":
    main()
