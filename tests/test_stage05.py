import unittest
import numpy as np
import pandas as pd
from pipeline.evaluate_stage05 import agreement_rows, summarize_agreement, provenance_labels


class Stage05Tests(unittest.TestCase):
    def test_partition_estimand_uses_all_cross_and_unique_within_pairs(self):
        a = np.tile(np.repeat(np.arange(3), 3), (10, 1))
        rows = agreement_rows(a, a)
        self.assertEqual([sum(r['component'] == role for r in rows) for role in ('C','W_EN','W_VI')], [100,45,45])
        summary = summarize_agreement(rows)
        self.assertAlmostEqual(summary['C'], 1)
        self.assertAlmostEqual(summary['S'], 0)

    def test_s_is_signed_gap_not_absolute_ratio(self):
        summary = summarize_agreement([{'component':role,'ami':value} for role,value in [('C',.2),('W_EN',.8),('W_VI',.6)]])
        self.assertAlmostEqual(summary['S'], -.5)

    def test_provenance_does_not_claim_human_authorship(self):
        frame = pd.DataFrame({'text_source':['machine','human','unknown',None]})
        self.assertEqual(provenance_labels(frame).tolist(), ['translation-flagged','not-flagged','unknown','unknown'])


if __name__ == '__main__':
    unittest.main()
