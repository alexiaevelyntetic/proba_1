import unittest

import pandas as pd

from src.risk import is_catch_all


class CatchAllTest(unittest.TestCase):
    def test_flags_assortment_names_only(self):
        names = pd.Series(["PROMO WK33", "New lines - pending Q4", "Misc. (old)", "Feature space Spring",
                           "other juic", "BUT-Seasonal", "Crsps/Crisps-Old", "Newcastle Brown"])
        self.assertEqual(is_catch_all(names).tolist(), [True, True, True, True, False, False, False, False])


if __name__ == "__main__":
    unittest.main()
