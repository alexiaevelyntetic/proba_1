"""Produce the deliverable: one line per raw supplier_category value.

Settings come from grouped CV (`python -m src.evaluate`): the product name without the
category name classifies products best, and hand labels only nudge the summed model
probabilities (heavier label weights did worse in CV). Products are counted equally:
revenue weighting scored the same in CV but let a few big sellers overrule the bulk of a
category (e.g. tissues inside 'PR KITCHEN-ROLL'); sales are used to rank risk instead.
Run with `python -m src.map`.
"""
import numpy as np
import pandas as pd

from src.data import DATA_DIR, load
from src.model import ProductClassifier, aggregate, combine, label_votes

C = 10.0
USE_CATEGORY_NAME = False
REVENUE_WEIGHT = 0.0
LABEL_WEIGHT = 0.25

OUT_DIR = DATA_DIR / "output"


def build_mapping(products, labels, categories, return_products=False):
    """Return one row per folded category key with the chosen id and diagnostics.

    With return_products, also return per-product `p_chosen` (the model's probability that
    the product belongs in the category its supplier category was mapped to) and `pred_id`
    (the model's own best category for the product).
    """
    labelled = labels.merge(products, on="supplier_sku")
    all_classes = sorted(categories.our_category_id)

    model = ProductClassifier(C=C, use_category_name=USE_CATEGORY_NAME).fit(labelled, labelled.our_category_id)
    proba = model.predict_proba(products)
    model_scores = aggregate(products, proba, model.classes_, REVENUE_WEIGHT).reindex(columns=all_classes, fill_value=0)

    n_labels = labelled.groupby("key").size()
    scores = combine(model_scores, label_votes(labelled, all_classes), n_labels, LABEL_WEIGHT)
    scores = scores.div(scores.sum(axis=1), axis=0)

    values = scores.to_numpy()
    order = np.argsort(-values, axis=1)
    classes = np.asarray(all_classes)
    rows = np.arange(len(values))
    out = pd.DataFrame({
        "key": scores.index,
        "our_category_id": classes[order[:, 0]],
        "confidence": values[rows, order[:, 0]],
        "runner_up_id": classes[order[:, 1]],
        "runner_up_score": values[rows, order[:, 1]],
        "model_only_id": classes[model_scores.loc[scores.index].to_numpy().argmax(1)],
    })
    majority = labelled.groupby("key").our_category_id.agg(lambda s: s.value_counts().index[0])
    out["n_labels"] = out.key.map(n_labels).fillna(0).astype(int)
    out["label_majority_id"] = out.key.map(majority)

    # Share of the category's products the model would place in the chosen category:
    # a low value means the supplier category is mixed and any single answer misplaces some.
    product_pred = pd.Series(model.classes_[proba.argmax(1)], index=products.index)
    agree = (product_pred == products.key.map(out.set_index("key").our_category_id))
    out["purity"] = out.key.map(agree.groupby(products.key).mean())
    stats = products.groupby("key").agg(n_products=("supplier_sku", "size"), revenue=("revenue", "sum"))
    out = out.merge(stats, left_on="key", right_index=True)
    if not return_products:
        return out
    chosen = products.key.map(out.set_index("key").our_category_id)
    col = {c: i for i, c in enumerate(model.classes_)}
    idx = chosen.map(col)
    p_chosen = np.where(idx.notna(), proba[np.arange(len(products)), idx.fillna(0).astype(int)], 0.0)
    return out, products.assign(our_category_id=chosen.to_numpy(), p_chosen=p_chosen,
                                pred_id=product_pred.to_numpy())


def main():
    products, labels, _, categories = load()
    mapping = build_mapping(products, labels, categories)

    raw = products[["supplier_category", "key"]].drop_duplicates().sort_values("supplier_category")
    detail = raw.merge(mapping, on="key", how="left")
    assert detail.our_category_id.notna().all()

    paths = dict(zip(categories.our_category_id, categories.path))
    detail["our_path"] = detail.our_category_id.map(paths)
    detail["runner_up_path"] = detail.runner_up_id.map(paths)

    OUT_DIR.mkdir(exist_ok=True)
    detail[["supplier_category", "our_category_id"]].to_csv(OUT_DIR / "mapping.csv", index=False)
    detail.to_csv(OUT_DIR / "mapping_detail.csv", index=False)

    folded = detail.drop_duplicates("key")
    labelled = folded.n_labels > 0
    print(f"wrote {len(detail)} lines ({len(folded)} folded categories) to {OUT_DIR / 'mapping.csv'}")
    print(f"labelled categories where final answer differs from label majority: "
          f"{(folded[labelled].our_category_id != folded[labelled].label_majority_id).sum()} / {labelled.sum()}")
    print(f"our categories used: {folded.our_category_id.nunique()} / {len(categories)}")
    print(f"median confidence {folded.confidence.median():.2f}, median purity {folded.purity.median():.2f}")


if __name__ == "__main__":
    main()
