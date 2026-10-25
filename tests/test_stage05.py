import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
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

    def test_probe_selection_does_not_read_test_labels(self):
        from pipeline.probe_stage05 import fit_select
        rng = np.random.default_rng(42)
        x = rng.normal(size=(36,4))
        y = np.tile(np.arange(3),12)
        split = np.repeat(['train','validation','test'],12)
        _, pred1, a = fit_select(x,y,split,[0,1,2])
        changed = y.copy()
        changed[split=='test'] = (changed[split=='test']+1)%3
        _, pred2, b = fit_select(x,changed,split,[0,1,2])
        self.assertEqual(a['tuning'],b['tuning'])
        self.assertEqual(a['selected_C'],b['selected_C'])
        np.testing.assert_array_equal(pred1,pred2)

    def test_e5_identical_native_t512_still_produce_four_independent_cluster_calls(self):
        from pipeline import evaluate_stage05 as stage
        same = np.ones((3,4),dtype=np.float32)
        inputs = {'tn2:'+lang+':'+condition:same for lang in ('en','vi') for condition in ('native','token_budget')}
        frame = pd.DataFrame({'genreId':['a','b','c']})
        output = (np.tile(np.arange(3),(10,1)),np.ones(10),np.ones(10))
        with tempfile.TemporaryDirectory() as tmp, patch.object(stage,'load_validated_e5_tn2',return_value=(inputs,{'tn2':frame})), patch.object(stage.cluster,'run_kmeans_grid',return_value=output) as cluster_mock, patch.object(stage.metrics,'external_scores',return_value={'ami':0.,'ari':0.}), patch.object(stage,'evaluate_pair') as pair_mock:
            result = stage.run(Path(tmp),'e5_tn2')
            self.assertEqual(cluster_mock.call_count,4)
            self.assertEqual(pair_mock.call_count,2)
            self.assertEqual(result['retrieval_queries'],54628)
            self.assertEqual(len(list(Path(tmp).glob('*kmeans_main.npz'))),4)


if __name__ == '__main__':
    unittest.main()
