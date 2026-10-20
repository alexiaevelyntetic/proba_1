"""Grouped cross-validation: whole supplier categories are held out, as unlabelled ones are.

For a held-out category we report
  * model-only: the mapping we'd give a category with no labels at all;
  * combined:   leave-one-out within the category - the model plus the *other* labels,
                scored on the left-out label (directly comparable to the baseline).
Accuracy is per labelled product: did the product land where our team put it.
Run with `python -m src.evaluate`.
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.data import load
from src.model import ProductClassifier, aggregate

FOLDS = 5


def assign_folds(labelled, seed=0):
    keys = labelled.key.unique()
    group_of = dict(zip(keys, np.random.default_rng(seed).permutation(len(keys)) % FOLDS))
    return labelled.key.map(group_of)


def fold_scores(products, labelled, all_classes, C, use_category_name, revenue_weights, label_weights,
                fit=None):
    """Return a long DataFrame of per-label predictions for every (rw, lw) setting.

    `combine` adds label_weight * (count of each class among the labels); leaving one label
    out is therefore the same as subtracting its one-hot row, which keeps this vectorised.
    `fit(train_labelled, train_products) -> model` lets callers swap the training scheme.
    """
    col = {c: i for i, c in enumerate(all_classes)}
    folds = assign_folds(labelled)
    frames = []
    for f in range(FOLDS):
        train, test = labelled[folds != f], labelled[folds == f]
        held = products[products.key.isin(test.key)]
        if fit is None:
            model = ProductClassifier(C=C, use_category_name=use_category_name).fit(train, train.our_category_id)
        else:
            model = fit(train, products[~products.key.isin(test.key)])
        proba = model.predict_proba(held)
        prod_pred = pd.Series(model.classes_[proba.argmax(1)], index=held.supplier_sku.to_numpy())

        truth_idx = test.our_category_id.map(col).to_numpy()
        onehot = np.zeros((len(test), len(all_classes)))
        onehot[np.arange(len(test)), truth_idx] = 1
        counts = pd.DataFrame(onehot, index=test.key.to_numpy()).groupby(level=0).sum()
        key_counts = counts.loc[test.key].to_numpy()
        n_other = key_counts.sum(1) - 1
        base = dict(key=test.key.to_numpy(), sku=test.supplier_sku.to_numpy(),
                    truth=test.our_category_id.to_numpy(), revenue=test.revenue.to_numpy(),
                    n_other=n_other, product_pred=prod_pred[test.supplier_sku].to_numpy())
        for rw in revenue_weights:
            agg = aggregate(held, proba, model.classes_, revenue_weight=rw).reindex(columns=all_classes, fill_value=0)
            scores = agg.loc[test.key].to_numpy()
            model_pred = np.asarray(all_classes)[scores.argmax(1)]
            for lw in label_weights:
                comb = scores + lw * (key_counts - onehot)
                frames.append(pd.DataFrame(dict(base, rw=rw, lw=lw, model_pred=model_pred,
                                                comb_pred=np.asarray(all_classes)[comb.argmax(1)])))
    return pd.concat(frames, ignore_index=True)


def summarise(df):
    acc = lambda pred, m=None: (df.truth == pred)[m].mean() if m is not None else (df.truth == pred).mean()
    wacc = lambda pred, m: ((df.truth == pred) * df.revenue)[m].sum() / df.revenue[m].sum()
    multi = df.n_other > 0
    return pd.Series({
        "product_acc": acc(df.product_pred),
        "model_only": acc(df.model_pred),
        "model_only_rev": wacc(df.model_pred, slice(None)),
        "combined_loo": acc(df.comb_pred, multi),
        "combined_loo_rev": wacc(df.comb_pred, multi),
    })


def main():
    products, labels, _, categories = load()
    labelled = labels.merge(products, on="supplier_sku")
    all_classes = sorted(categories.our_category_id)
    results = []
    for C, ucn in itertools.product([3.0, 10.0, 30.0], [True, False]):
        df = fold_scores(products, labelled, all_classes, C, ucn,
                         revenue_weights=[0.0, 0.5], label_weights=[0.25, 0.5, 1.0, 2.0])
        for (rw, lw), sub in df.groupby(["rw", "lw"]):
            results.append({"C": C, "cat_name": ucn, "rev_w": rw, "label_w": lw, **summarise(sub)})
        print(f"done C={C} cat_name={ucn}", flush=True)
    res = pd.DataFrame(results).sort_values("combined_loo", ascending=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False, float_format="{:.3f}".format))
    print("\nbaseline (labels only) leave-one-out: 0.808, revenue-weighted 0.824")


if __name__ == "__main__":
    main()
