"""Step 4: product-level classifier, aggregated to one answer per supplier category.

Each product is classified from its name (brand removed: brands span many categories and
carry no signal), its supplier category name, and its pack size: the kind of unit plus a
size band, because our tree splits catering from retail packs at 5 kg / 5 l. The
per-product probabilities are summed over every product in a supplier category, and any
hand labels in that category are added as extra votes.
"""
import re

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

UNIT_KIND = {
    "kg": "weight", "g": "weight", "gr": "weight",
    "l": "volume", "ml": "volume", "cl": "volume", "ltr": "volume", "litre": "volume",
    "pk": "count", "pcs": "count", "pack": "count", "each": "count", "pc": "count",
    "ea": "count", "single": "count",
}


TO_BASE = {"kg": 1, "g": 1e-3, "gr": 1e-3, "l": 1, "ml": 1e-3, "cl": 1e-2, "ltr": 1, "litre": 1}
PACK = re.compile(r"^\s*(?:(\d+)\s*x\s*)?(\d+(?:[.,]\d+)?)\s*([a-z]+)\s*$")


def unit_kind(pack_size: str) -> str:
    unit = re.sub(r"[\d.,x\s]+", "", pack_size.lower())
    if not unit and re.fullmatch(r"\s*x\s*\d+\s*", pack_size.lower()):
        return "count"  # 'x12'
    return UNIT_KIND.get(unit, "none")


def pack_amount(pack_size: str):
    """Total kg or litres in the pack ('2x300 gr' -> 0.6), or None for counts and unknowns."""
    m = PACK.match(pack_size.lower())
    if not m or m.group(3) not in TO_BASE:
        return None
    return int(m.group(1) or 1) * float(m.group(2).replace(",", ".")) * TO_BASE[m.group(3)]


def pack_tokens(pack_size: str) -> str:
    """Unit kind plus size band. Our tree splits catering from retail packs at 5 kg / 5 l."""
    kind = unit_kind(pack_size)
    amount = pack_amount(pack_size)
    if amount is None:
        return kind
    band = "lt0.5" if amount < 0.5 else "lt2" if amount < 2 else "lt5" if amount < 5 else "ge5"
    return f"{kind} {kind}_{band} {'catering' if amount >= 5 else 'retail'}"


def name_text(products: pd.DataFrame) -> pd.Series:
    def clean(name, brand):
        name = name.lower()
        if brand:
            name = name.replace(brand.lower(), " ")
        return re.sub(r"\s+", " ", name).strip()
    return pd.Series([clean(n, b) for n, b in zip(products.product_name, products.brand)],
                     index=products.index)


def category_text(products: pd.DataFrame) -> pd.Series:
    return products.supplier_category.str.lower().str.replace(r"[^a-z]+", " ", regex=True).str.strip()


class ProductClassifier:
    def __init__(self, C: float = 10.0, use_category_name: bool = True, use_pack_amount: bool = True,
                 pack_weight: float = 1.0):
        self.C = C
        self.use_category_name = use_category_name
        self.use_pack_amount = use_pack_amount
        self.pack_weight = pack_weight

    def _features(self, products, fit=False):
        pack = products.pack_size.map(pack_tokens if self.use_pack_amount else unit_kind)
        parts = [name_text(products), category_text(products), pack]
        if fit:
            self.vecs = [
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2),
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2),
                TfidfVectorizer(analyzer="word", token_pattern=r"\S+"),
            ]
            mats = [v.fit_transform(t) for v, t in zip(self.vecs, parts)]
        else:
            mats = [v.transform(t) for v, t in zip(self.vecs, parts)]
        mats[2] = mats[2] * self.pack_weight
        if not self.use_category_name:
            mats.pop(1)
        return hstack(mats).tocsr()

    def fit(self, products: pd.DataFrame, target: pd.Series, sample_weight=None):
        X = self._features(products, fit=True)
        self.clf = LogisticRegression(C=self.C, max_iter=2000)
        self.clf.fit(X, target, sample_weight=sample_weight)
        self.classes_ = self.clf.classes_
        return self

    def predict_proba(self, products: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self._features(products))


def aggregate(products: pd.DataFrame, proba: np.ndarray, classes, revenue_weight: float = 0.0) -> pd.DataFrame:
    """Sum product probabilities per category -> DataFrame (key x class), rows sum to 1.

    revenue_weight blends count-weighting (0) towards revenue-weighting (1), so that when a
    category is mixed the answer leans towards where its money is.
    """
    rev = products.revenue.to_numpy(dtype=float)
    key_rev = products.groupby("key").revenue.transform("sum").to_numpy(dtype=float)
    share = np.divide(rev, key_rev, out=np.zeros_like(rev), where=key_rev > 0)
    n = products.groupby("key").key.transform("size").to_numpy(dtype=float)
    w = (1 - revenue_weight) / n + revenue_weight * share
    scores = pd.DataFrame(proba * w[:, None], columns=classes, index=products.index)
    scores["key"] = products.key.to_numpy()
    agg = scores.groupby("key").sum()
    return agg.div(agg.sum(axis=1), axis=0)


def label_votes(labelled: pd.DataFrame, classes) -> pd.DataFrame:
    """Share of hand labels per (key, class); rows sum to 1."""
    votes = pd.crosstab(labelled.key, labelled.our_category_id).reindex(columns=classes, fill_value=0)
    return votes.div(votes.sum(axis=1), axis=0)


def combine(model_scores: pd.DataFrame, votes: pd.DataFrame, n_labels: pd.Series, label_weight: float) -> pd.DataFrame:
    """Model distribution plus label shares; labels count more the more of them there are."""
    w = (label_weight * n_labels).reindex(model_scores.index).fillna(0)
    v = votes.reindex(index=model_scores.index, columns=model_scores.columns).fillna(0)
    return model_scores.add(v.mul(w, axis=0))
