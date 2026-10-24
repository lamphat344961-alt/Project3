"""Synthetic unit tests; these never claim a real-repo pilot PASS."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pipeline'))
import gate5_statistics as g

class Gate5UnitTests(unittest.TestCase):
    def test_percentile_inversion_and_tost(self):
        a=np.linspace(-.004,.004,1000)
        lo,hi=g.ci(a,.05)
        self.assertLess(lo,0);self.assertGreater(hi,0)
        self.assertAlmostEqual(g.percentile_p(a,0,'two'),1.)
        self.assertEqual(g.tost_p(a,.01),0.)
        self.assertGreaterEqual(g.tost_p(a,.003),0.)
        positive=np.linspace(.02,.03,1000)
        self.assertLess(g.percentile_p(positive,0,'greater'),.05)
        self.assertEqual(g.percentile_p(positive,0,'two'),0.)
        self.assertGreater(g.tost_p(positive,.01),.05)
        self.assertTrue(g.ci(a,.1)[0]>-.01 and g.ci(a,.1)[1]<.01)
    def test_holm_stepdown_family(self):
        p=[.011,.001,.03,.8,.009,.025,.004,.02]
        q=g.holm(p)
        self.assertEqual(len(q),8)
        self.assertTrue((q>=p).all())
        order=np.argsort(p)
        self.assertTrue((np.diff(q[order])>=0).all())
        self.assertAlmostEqual(q[order[0]],.008)
    def test_bootstrap_fixed_partition_by_app_and_checkpoint_resume(self):
        class Input:
            pass
        obj=Input()
        obj.y={'tn1':np.array([0,0,1,1,2,2]),'tn2':np.array([0,0,1,1,2,2])}
        obj.labels={}
        labels=np.array([0,1,0,1,2,2],dtype=np.int32)
        for rep in g.REPS1:
            for tr in g.TR1:
                obj.labels[g.key('tn1',rep,'vi',tr)]=np.stack([np.roll(labels,i%3) for i in range(10)])
        for rep in g.REPS2:
            for tr in g.TR2:
                for lang in ('en','vi'):
                    obj.labels[g.key('tn2',rep,lang,tr)]=np.stack([np.roll(labels,i%3) for i in range(10)])
        ii=np.stack([np.arange(6),np.array([0,0,1,2,4,5]),np.array([1,2,3,3,4,5])])
        class Args:tn1_block=2;tn2_block=2
        with tempfile.TemporaryDirectory() as d,patch.object(g,'B',3):
            out=Path(d)
            only=[('tn1',g.REPS1[0],'native'),('tn2',g.REPS2[0],'native')]
            g.run_blocks(out,obj,{'tn1':ii,'tn2':ii},Args(),configs=only)
            with self.assertRaisesRegex(RuntimeError,'INCOMPLETE BLOCK'):
                g.load_blocks(out,'tn1',g.REPS1[-1],'native',2)
            a=g.load_blocks(out,'tn1',g.REPS1[0],'native',2)
            self.assertEqual(a['ami_vi'].shape,(3,10))
            self.assertAlmostEqual(a['ami_vi'][0,0],g.ami(obj.y['tn1'],labels))
            z=g.load_blocks(out,'tn2',g.REPS2[0],'native',2)
            self.assertEqual(z['cross'].shape,(3,10,10))
            self.assertEqual(z['within_en'].shape,(3,45))
            self.assertEqual(z['within_vi'].shape,(3,45))
            count=len(list((out/'blocks').glob('*.npz')))
            g.run_blocks(out,obj,{'tn1':ii,'tn2':ii},Args(),configs=only)
            self.assertEqual(count,len(list((out/'blocks').glob('*.npz'))))
            self.assertAlmostEqual(z['cross'][0,1,0],g.ami(np.roll(labels,1),labels))
            remaining=[c for c in g.ALL_CONFIGS if c not in only]
            g.run_blocks(out,obj,{'tn1':ii,'tn2':ii},Args(),configs=remaining)
            self.assertEqual(len(list((out/'blocks').glob('*.npz'))),len(g.ALL_CONFIGS)*2)
            self.assertEqual(g.load_blocks(out,'tn1',g.REPS1[-1],'native',2)['ami_vi'].shape,(3,10))
    def test_same_sampling_indices_and_seed_weights(self):
        # Tiny B but still full canonical row widths: construction is deterministic.
        with tempfile.TemporaryDirectory() as d, patch.object(g,'B',3):
            out=Path(d)
            g.create_rng_files(out)
            sig=g.rng_fingerprints(out)
            sample,weights=g.load_rng(out,sig)
            h={name:g.sha(out/f'indices_{name}.npy') for name in ('tn1','tn2')}
            try:
                self.assertEqual(sample['tn1'].shape,(3,23883));self.assertEqual(sample['tn2'].shape,(3,13657))
                self.assertEqual(len(weights),32)
                self.assertTrue(all(np.all(w.sum(axis=1)==10) for w in weights.values()))
                a=('tn2','bge_m3','native','en');b=('tn2','bge_m3','native','vi')
                self.assertFalse(np.array_equal(weights[a],weights[b]))
                g.create_rng_files(out)
                sample2,weights2=g.load_rng(out,sig)
                try:
                    self.assertTrue(np.array_equal(sample['tn1'],sample2['tn1']))
                    self.assertEqual(h['tn1'],g.sha(out/'indices_tn1.npy'))
                    self.assertEqual(sig,g.rng_fingerprints(out))
                finally:g.close_mmaps(sample2,weights2)
            finally:g.close_mmaps(sample,weights)
            # Fresh directory must yield the exact same per-file RNG bytes.
            with tempfile.TemporaryDirectory() as other:
                g.create_rng_files(Path(other))
                self.assertEqual(sig,g.rng_fingerprints(Path(other)))
    def test_config_selection_and_checkpoints(self):
        self.assertEqual(len(g.ALL_CONFIGS),24)
        self.assertEqual(len(g.WEIGHT_KEYS),32)
        chosen=g.parse_configs(['tn1__bge_m3__native,tn2__tfidf_lsa__native'])
        self.assertEqual(chosen,(('tn1','bge_m3','native'),('tn2','tfidf_lsa','native')))
        with self.assertRaisesRegex(ValueError,'Unknown --configs'):
            g.parse_configs(['tn1__unknown__native'])
        with self.assertRaisesRegex(ValueError,'Duplicate config'):
            g.parse_configs(['tn1__bge_m3__native,tn1__bge_m3__native'])
    def test_tn2b_independent_language_seed_weights(self):
        # Cross uses EN(0,1) × VI(2,3), never the old same-weight square.
        we=np.zeros((2,10),dtype=int);wv=np.zeros((2,10),dtype=int)
        we[:,0]=4;we[:,1]=6
        wv[:,2]=3;wv[:,3]=7
        jj=[(i,j) for i in range(10) for j in range(i+1,10)]
        cross=np.tile(np.add.outer(np.arange(10)*100,np.arange(10)),(2,1,1)).astype(float)
        en=np.tile(np.array([i+j for i,j in jj],dtype=float),(2,1))
        vi=np.tile(np.array([2*i+j for i,j in jj],dtype=float),(2,1))
        d={'cross':cross,'within_en':en,'within_vi':vi}
        observed=g.tn2b_weighted_seed_sensitivity(d,we,wv)
        expected_cross=(4*3*2+4*7*3+6*3*102+6*7*103)/100
        enij=1 # only original EN pair (0,1) has nonzero weight
        viij=2*2+3 # only original VI pair (2,3)
        np.testing.assert_allclose(observed,expected_cross-(enij+viij)/2)
        we[:]=0;we[:,0]=10
        self.assertTrue(np.isnan(g.tn2b_weighted_seed_sensitivity(d,we,wv)).all())
    def test_npz_atomic_contents_and_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            dst=Path(d)/'checkpoint.npz'
            g.atomic_npz_new(dst,data=np.array([[3.,4.]]))
            with np.load(dst) as z:
                np.testing.assert_array_equal(z['data'],np.array([[3.,4.]]))
            self.assertFalse(dst.with_suffix('.partial.npz').exists())
            with self.assertRaises(FileExistsError):g.atomic_npz_new(dst,data=np.array([5.]))
            failed=Path(d)/'failed.npz'
            def fail_after_partial(handle,**arrays):
                handle.write(b'partial')
                raise OSError('simulated failed checkpoint write')
            with patch.object(g.np,'savez_compressed',side_effect=fail_after_partial):
                with self.assertRaisesRegex(OSError,'simulated'):
                    g.atomic_npz_new(failed,data=np.array([5.]))
            self.assertFalse(failed.exists())
            self.assertFalse(failed.with_suffix('.partial.npz').exists())
    def test_top_agglo_source_not_best_kmeans(self):
        import pandas as pd
        root=Path(__file__).resolve().parents[1]
        path=root/'results/evaluation/metrics_long.csv'
        if not path.exists(): path=Path(__file__).resolve().parents[2]/'metrics_long.csv'
        df=pd.read_csv(path) if path.exists() else None
        if df is None:self.skipTest('Real metric table not included with script-only package')
        x=df[(df.section=='TN1_EXTERNAL')&(df.algorithm=='agglomerative')&(df.metric=='ami')]
        best=x.sort_values('value',ascending=False).iloc[0]
        self.assertEqual((best.representation,best.truncation),('tfidf_lsa','char_budget'))

if __name__=='__main__':unittest.main()

class Gate5IntegrationSynthetic(unittest.TestCase):
    def test_finalize_shapes_all_52_comparisons(self):
        # Synthetic 1000-replicate checkpoint matrices, real output schema, no heavy AMI loop.
        import pandas as pd
        class Input:pass
        inp=Input()
        root=Path(__file__).resolve().parents[1]
        path=root/'results/evaluation/metrics_long.csv'
        if not path.exists(): path=Path(__file__).resolve().parents[2]/'metrics_long.csv'
        if not path.exists(): self.skipTest('Full Gate4 metric table not present')
        inp.df=pd.read_csv(path)
        inp.delta={(r,t):np.linspace(.12,.24,13657) for r in g.REPS2 for t in g.TR2}
        gen=np.random.default_rng(99)
        class Args:tn1_block=1000;tn2_block=1000
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'blocks').mkdir()
            np.save(out/'indices_tn2.npy',gen.integers(0,13657,(1000,13657),dtype=np.int32))
            for r in g.REPS1:
                for t in g.TR1:
                    a=gen.uniform(.1,.6,(1000,10))
                    g.atomic_npz_new(g.block_file(out,'tn1',r,t,0,1000),replicate=np.arange(1000),ami_vi=a)
            for r in g.REPS2:
                for t in g.TR2:
                    z={'replicate':np.arange(1000),
                        'ami_en':gen.uniform(.1,.6,(1000,10)),
                        'ami_vi':gen.uniform(.1,.6,(1000,10)),
                        'cross':gen.uniform(.4,.7,(1000,10,10)),
                        'within_en':gen.uniform(.4,.7,(1000,45)),
                        'within_vi':gen.uniform(.4,.7,(1000,45))}
                    g.atomic_npz_new(g.block_file(out,'tn2',r,t,0,1000),**z)
            w={k:np.ones((1000,10),dtype=np.int8) for k in g.WEIGHT_KEYS}
            summary,seedwise,reps=g.summarize(out,inp,w,Args())
            self.assertEqual(len(summary),52)
            self.assertEqual(len(seedwise),320000)
            self.assertEqual(len(reps),52000)
            self.assertEqual(set(summary.family),{'TN1-A','TN1-B','TOP3','TN2-A','TN2-B','TN2-C'})
            self.assertEqual(int(summary.holm_difference_p.notna().sum()),32)
            self.assertEqual(int(summary.holm_tost_p.notna().sum()),16)
            self.assertEqual(set(summary.status),{'VALID'})
