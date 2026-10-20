import unittest

from src.model import pack_amount, pack_tokens


class PackSizeTest(unittest.TestCase):
    def test_amount_in_kg_or_litres(self):
        self.assertAlmostEqual(pack_amount("2x300 gr"), 0.6)
        self.assertAlmostEqual(pack_amount("0,1 kg"), 0.1)
        self.assertAlmostEqual(pack_amount("12,5kg"), 12.5)
        self.assertAlmostEqual(pack_amount("200cl"), 2.0)
        self.assertAlmostEqual(pack_amount("5LTR"), 5.0)
        self.assertIsNone(pack_amount("24pk"))
        self.assertIsNone(pack_amount("single"))

    def test_catering_threshold_is_5(self):
        self.assertIn("catering", pack_tokens("5000g").split())
        self.assertIn("catering", pack_tokens("5 l").split())
        self.assertIn("retail", pack_tokens("4999 ml").split())
        self.assertEqual(pack_tokens("x12"), "count")


if __name__ == "__main__":
    unittest.main()
