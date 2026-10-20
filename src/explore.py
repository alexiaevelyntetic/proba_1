"""Step 1: data checks. Prints a short report; run with `python -m src.explore`."""
import pandas as pd

from src.data import load


def main():
    products, labels, sales, categories = load()
    cat_ids = set(categories["our_category_id"])
    lab = labels.merge(products, on="supplier_sku", how="left")

    print("== Sizes")
    print(f"products {len(products)}, labelled {len(labels)}, sales {len(sales)}, our categories {len(categories)}")

    print("\n== Integrity")
    print("SKU unique (products/labels/sales):",
          products.supplier_sku.is_unique, labels.supplier_sku.is_unique, sales.supplier_sku.is_unique)
    print("labelled SKUs missing from products:", lab["key"].isna().sum())
    print("label ids not in our_categories:", (~labels.our_category_id.isin(cat_ids)).sum())
    print("products without a sales row:", (~products.supplier_sku.isin(sales.supplier_sku)).sum())
    print("products with zero revenue:", (products.revenue <= 0).sum())
    print("our categories never used in labels:",
          sorted(cat_ids - set(labels.our_category_id)))

    print("\n== Supplier categories")
    spellings = products.groupby("key")["supplier_category"].nunique()
    print(f"distinct raw values {products.supplier_category.nunique()}, after case/space folding {len(spellings)}")
    print(f"folded categories with >1 spelling: {(spellings > 1).sum()}")
    for key in spellings[spellings > 1].index[:5]:
        print("  ", sorted(products.loc[products.key == key, "supplier_category"].unique()))
    size = products.groupby("key").size()
    print(f"products per category: min {size.min()}, median {size.median():.0f}, max {size.max()}")

    print("\n== Label coverage")
    per_key = lab.groupby("key").size()
    covered = products.key.isin(per_key.index)
    print(f"categories with >=1 label: {len(per_key)} / {len(size)}")
    print(f"labels per labelled category: median {per_key.median():.0f}, mean {per_key.mean():.1f}, "
          f"single-label {(per_key == 1).sum()}")
    print(f"revenue in labelled categories: {products.revenue[covered].sum() / products.revenue.sum():.1%}")

    purity = lab.groupby("key")["our_category_id"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    print(f"labelled categories whose labels disagree: {(purity < 1).sum()} "
          f"(of those with >=2 labels: {(per_key >= 2).sum()})")

    names = products.groupby("key")["supplier_category"].first()
    print("\nMost mixed labelled categories:")
    for key in purity.sort_values().index[:8]:
        print(f"  {names[key]!r}: purity {purity[key]:.2f}, {per_key[key]} labels")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
