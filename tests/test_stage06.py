import unittest
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score,adjusted_rand_score
from pipeline.ami_cached import expected_cell_table,score_pair


class ExactCachedAMITests(unittest.TestCase):
    def check_pair(self,a,b,k):
        counts=np.r_[np.bincount(a,minlength=k),np.bincount(b,minlength=k)]
        table=expected_cell_table(len(a),counts)
        ami,ari=score_pair(a,b,table,k=k)
        self.assertAlmostEqual(ami,adjusted_mutual_info_score(a,b),delta=1e-8)
        self.assertAlmostEqual(ari,adjusted_rand_score(a,b),delta=1e-10)

    def test_random_imbalanced_and_missing_labels_match_sklearn(self):
        rng=np.random.Generator(np.random.PCG64(42))
        for n,k in [(6,3),(49,49),(400,9),(1000,49)]:
            for _ in range(3):
                a=rng.integers(0,k,n)
                b=rng.integers(0,k,n)
                self.check_pair(a,b,k)

    def test_perfect_constant_singleton_and_one_constant_cases(self):
        for a,b,k in [(np.zeros(8,dtype=int),np.ones(8,dtype=int),2),
                      (np.zeros(8,dtype=int),np.arange(8)%2,2),
                      (np.arange(8),np.arange(8)[::-1],8),
                      (np.arange(8),np.arange(8)%3,8),
                      (np.array([0,0,1,1]),np.array([1,1,0,0]),2)]:
            self.check_pair(a,b,k)

    def test_uncovered_marginals_fail_instead_of_approximating(self):
        table=expected_cell_table(8,[2,6])
        with self.assertRaisesRegex(ValueError,'lookup'):
            score_pair(np.arange(8)%2,np.array([0,0,0,1,1,1,1,1]),table,k=2)

    def test_empty_draw_retained_undefined(self):
        a,b=score_pair(np.array([],dtype=int),np.array([],dtype=int),np.zeros((1,1)))
        self.assertTrue(np.isnan(a) and np.isnan(b))


class N05DesignTests(unittest.TestCase):
    def test_without_replacement_shared_deterministic_subsets(self):
        from pipeline.statistics_stage06 import make_subsets
        a=make_subsets({'tn1':20,'tn2':30},b=3)
        b=make_subsets({'tn1':20,'tn2':30},b=3)
        for exp,n in [('tn1',20),('tn2',30)]:
            np.testing.assert_array_equal(a[exp],b[exp])
            self.assertEqual(a[exp].shape,(3,int(.8*n)))
            self.assertTrue(all(len(np.unique(draw))==len(draw) and np.all(np.diff(draw)>0) for draw in a[exp]))

    def test_undefined_draw_counts_and_rank_ties_are_retained(self):
        import pandas as pd
        from pipeline.statistics_stage06 import summaries,rank_summary
        result=summaries([1.,np.nan,3.])
        self.assertEqual((result['n_defined'],result['n_undefined']),(2,1))
        meta=pd.DataFrame({'group':['x','x'],'representation':['a','b']})
        rows=rank_summary(np.ones((3,2)),np.ones(2),meta,['group'],'ami')
        self.assertTrue(all(r['rank_mean']==1.5 and r['share_draws_top_tie_adjusted']==.5 for r in rows))
        rows=rank_summary(np.full((3,2),np.nan),np.ones(2),meta,['group'],'ami')
        self.assertTrue(all(r['n_defined']==0 and r['n_undefined']==3 and np.isnan(r['rank_mean']) for r in rows))


if __name__=='__main__':
    unittest.main()
