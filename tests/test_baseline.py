import unittest

import pandas as pd

from src.baseline import vote
from src.data import category_key, load


class CategoryKeyTest(unittest.TestCase):
    def test_folds_case_and_whitespace(self):
        self.assertEqual(category_key("Apls   pear"), category_key("APLS PEAR"))
        self.assertEqual(category_key("bab fd"), category_key("Babfd"))
        self.assertEqual(category_key("A\tB c"), "abc")

    def test_keeps_punctuation(self):
        self.assertNotEqual(category_key("crsps:cra"), category_key("crsps cra"))


class VoteTest(unittest.TestCase):
    def test_majority_then_revenue_then_id(self):
        df = pd.DataFrame({
            "key": ["a", "a", "a", "b", "b", "c", "c"],
            "our_category_id": ["1", "1", "2", "3", "4", "6", "5"],
            "revenue": [1, 1, 100, 5, 9, 1, 1],
        })
        got = vote(df).set_index("key")["our_category_id"].to_dict()
        self.assertEqual(got, {"a": "1", "b": "4", "c": "5"})


class BaselineOnRealDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.products, cls.labels, _, cls.categories = load()

    def test_same_folded_name_gets_same_answer(self):
        lab = self.labels.merge(self.products[["supplier_sku", "key", "revenue"]], on="supplier_sku")
        mapping = vote(lab).set_index("key")["our_category_id"]
        raw = self.products[["supplier_category", "key"]].drop_duplicates()
        raw["pred"] = raw["key"].map(mapping)
        self.assertTrue((raw.dropna().groupby("key")["pred"].nunique() == 1).all())
        self.assertTrue(raw["pred"].dropna().isin(self.categories["our_category_id"]).all())


if __name__ == "__main__":
    unittest.main()
