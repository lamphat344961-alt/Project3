import unittest
import numpy as np
from pipeline.stage05_remaining import lexical_features


class LexicalTrainOnlyTests(unittest.TestCase):
    def test_heldout_vocabulary_never_enters_fitted_transform(self):
        spec={'vectorizer':{'analyzer':'word','ngram_range':[1,1],'min_df':1,'max_df':1.0,'sublinear_tf':True,'norm':'l2','lowercase':True,'token_pattern':r'(?u)\b\w\w+\b'},'svd':{'n_components':2,'random_state':42}}
        texts=['alpha apple common','beta banana common','alpha banana common','beta apple common','heldoutmarker apple','heldoutmarker banana']
        result,vec,svd=lexical_features([texts],spec,[np.array([1,1,1,1,0,0],dtype=bool)])
        self.assertNotIn('heldoutmarker',vec.vocabulary_)
        self.assertEqual(result[0].shape,(6,2))
        self.assertTrue(np.isfinite(result[0]).all())
        self.assertEqual(svd.n_components,2)


if __name__=='__main__':
    unittest.main()
