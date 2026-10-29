import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import m2


class UnitTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(m2.normalize_arabic(' أَإِآٱـ  بٌ\u200f '), 'اااا ب')

    def test_idempotent(self):
        t = 'إِنَّ اللَّيْلَ طَوِيلٌ'
        self.assertEqual(m2.normalize_arabic(t), m2.normalize_arabic(m2.normalize_arabic(t)))

    def test_structure_markers(self):
        p = {'verses': [['من كان', 'الى دار'], ['في ليل', 'عن قوم']]}
        self.assertEqual(m2.poem_to_sequence(p, False), 'من كان|الى دار#في ليل|عن قوم')

    def test_rhyme_removal_only_last_word(self):
        p = {'verses': [['من كان', 'الى دار']]}
        self.assertEqual(m2.poem_to_sequence(p, True), 'من كان|الى')

    def test_rhyme_fallback(self):
        p = {'verses': [['من كان', '']]}
        self.assertEqual(m2.poem_to_sequence(p, True), 'من|')

    def test_fragmentation(self):
        p = dict(verses=[['من كان', 'الى دار']]*23, author='a')
        texts, y, owners = m2.make_training([p])
        self.assertEqual([t.count('#')+1 for t in texts], [10,10,3])
        self.assertTrue((owners == 0).all())
        self.assertEqual(list(y), ['a']*3)

    def test_conformal_ties(self):
        p = m2.conformal_pvalues(np.array([[0.8]]), ['a'], {'a':[0.2,0.4]})
        self.assertEqual(float(p[0,0]), 1.0)

    def test_conformal_minimum_nonzero(self):
        p = m2.conformal_pvalues(np.array([[0.0]]), ['a'], {'a':[0.1,0.2]})
        self.assertEqual(float(p[0,0]), 1/3)

    def test_empty_conformal_class_errors(self):
        with self.assertRaises(ValueError):
            m2.conformal_pvalues(np.array([[0.5]]), ['a'], {'a':[]})

    def test_holm(self):
        self.assertTrue(np.allclose(m2.holm_adjust([0.01,0.03,0.2]), [.03,.06,.2]))


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).parent
        cls.r = cls.root/'results'
        cls.bundle = joblib.load(cls.r/'m2_model.joblib')
        cls.manifest = pd.read_csv(cls.r/'split_manifest.csv')
        cls.metrics = json.loads((cls.r/'metrics.json').read_text())
        cls.meta = json.loads((cls.r/'metadata.json').read_text())
        cls.pred = pd.read_csv(cls.r/'test_predictions.csv')

    def test_no_key_leakage(self):
        self.assertTrue(self.manifest.key.is_unique)
        self.assertEqual(len(self.manifest), self.meta['retained_poems'])

    def test_raw_data_no_shared_beyts(self):
        source = self.root/'data'
        if not source.exists():
            source = self.root.parent.parent
        poems = m2.load_corpus(source)
        kept, _ = m2.clean_corpus(poems)
        seen = {}
        for p in kept:
            for v in set(p['signature'].split('#')):
                self.assertNotIn(v, seen)
                seen[v] = p['key']

    def test_classes_all_roles(self):
        for role, rows in self.manifest.groupby('role'):
            self.assertEqual(sorted(rows.author.unique()), self.bundle['classes'])

    def test_probability_simplex(self):
        q = self.pred[[f'prob__{a}' for a in self.bundle['classes']]].values
        self.assertTrue(np.allclose(q.sum(axis=1),1))
        self.assertTrue(((q>=0)&(q<=1)).all())

    def test_p_values_bounds_and_resolution(self):
        for a in self.bundle['classes']:
            p = self.pred[f'p__{a}'].values
            n = len(self.bundle['conformal_by_class'][a])
            self.assertTrue((p>=1/(n+1)-1e-10).all())
            self.assertTrue((p<=1).all())
            self.assertTrue(np.allclose(p*(n+1),np.rint(p*(n+1))))

    def test_prediction_counts(self):
        self.assertEqual(self.metrics['correct'],int(self.pred.correct.sum()))
        self.assertEqual(self.metrics['n_poems'],len(self.pred))

    def test_permutation_plus_one(self):
        z = self.metrics['permutation_test']
        self.assertEqual(z['p_value'],(z['n_scores_ge_observed']+1)/(z['n_permutations']+1))
        self.assertGreater(z['p_value'],0)

    def test_mualaka_excluded_from_reference(self):
        keys = set(self.manifest.loc[self.manifest.role=='mualaka_diagnostic','key'])
        self.assertFalse(keys & set(self.bundle['all_reference_keys']))

    def test_claims_not_probability(self):
        self.assertFalse(np.allclose(self.pred.p_claimed,self.pred.probability_claimed))

    def test_external_prediction_actual_test_poem(self):
        source = self.root/'data'
        if not source.exists(): source = self.root.parent.parent
        poems = m2.load_corpus(source)
        key = self.pred.iloc[0]['key']
        p = next(p for p in poems if p['key']==key)
        with tempfile.TemporaryDirectory() as td:
            inp=Path(td)/'query.json';out=Path(td)/'result.json'
            inp.write_text(json.dumps(dict(p,claimed_author=p['author'])),encoding='utf-8')
            m2.predict(self.r/'m2_model.joblib',inp,out)
            z=json.loads(out.read_text())
            self.assertAlmostEqual(z['p_claimed'],self.pred.iloc[0]['p_claimed'])

    def test_inference_refuses_reference_overlap(self):
        beyt = self.bundle['all_reference_beyts'][0]
        with tempfile.TemporaryDirectory() as td:
            inp=Path(td)/'query.json'
            inp.write_text(json.dumps(dict(verses=[beyt.split('|')],claimed_author=self.bundle['classes'][0])))
            with self.assertRaises(ValueError):
                m2.predict(self.r/'m2_model.joblib',inp,Path(td)/'result.json')


if __name__=='__main__':
    unittest.main(verbosity=2)
