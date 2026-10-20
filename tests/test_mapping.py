import unittest

import pandas as pd

from src.data import category_key, load
from src.map import build_mapping
from src.model import aggregate


class AggregateTest(unittest.TestCase):
    def test_count_weighted_majority(self):
        products = pd.DataFrame({"key": ["a", "a", "a"], "revenue": [1.0, 1.0, 1000.0]})
        proba = [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]]
        self.assertEqual(aggregate(products, pd.DataFrame(proba).to_numpy(), ["x", "y"]).idxmax(axis=1)["a"], "x")
        by_rev = aggregate(products, pd.DataFrame(proba).to_numpy(), ["x", "y"], revenue_weight=1.0)
        self.assertEqual(by_rev.idxmax(axis=1)["a"], "y")


class MappingOutputTest(unittest.TestCase):
    """End-to-end on the real data: the file we hand over meets the brief's format rules."""

    @classmethod
    def setUpClass(cls):
        cls.products, labels, _, cls.categories = load()
        mapping = build_mapping(cls.products, labels, cls.categories).set_index("key")
        raw = cls.products[["supplier_category"]].drop_duplicates()
        raw["our_category_id"] = raw.supplier_category.map(category_key).map(mapping.our_category_id)
        cls.out = raw

    def test_one_line_per_distinct_raw_value(self):
        self.assertEqual(len(self.out), self.products.supplier_category.nunique())
        self.assertTrue(self.out.supplier_category.is_unique)

    def test_every_value_mapped_to_a_real_category(self):
        self.assertTrue(self.out.our_category_id.notna().all())
        self.assertTrue(self.out.our_category_id.isin(self.categories.our_category_id).all())

    def test_case_and_space_variants_agree(self):
        per_key = self.out.groupby(self.out.supplier_category.map(category_key)).our_category_id.nunique()
        self.assertTrue((per_key == 1).all())


if __name__ == "__main__":
    unittest.main()
