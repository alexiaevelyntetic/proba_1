"""Loading the input CSVs and the category-name normalisation the supplier uses."""
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent


def category_key(name: str) -> str:
    """Supplier treats names that differ only in case or whitespace as the same category."""
    return re.sub(r"\s+", "", name).lower()


def load(data_dir: Path = DATA_DIR):
    """Return (products, labels, sales, categories). Products carry `key` and sales columns."""
    read = lambda f, **kw: pd.read_csv(data_dir / f, dtype=str, keep_default_na=False, **kw)
    products = read("supplier_products.csv")
    labels = read("labelled_products.csv")
    categories = read("our_categories.csv")
    sales = pd.read_csv(data_dir / "sales.csv", dtype={"supplier_sku": str})

    products["key"] = products["supplier_category"].map(category_key)
    products = products.merge(sales, on="supplier_sku", how="left")
    products[["units", "revenue"]] = products[["units", "revenue"]].fillna(0)
    return products, labels, sales, categories
