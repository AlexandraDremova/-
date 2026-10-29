from __future__ import annotations
import argparse
import hashlib
import json
import re
import unicodedata
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, log_loss)
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

SEED = 20261028
CONFIG = dict(C=0.1, fragment_beyts=10, remove_rhyme=True,
              ngram_range=(1, 4), min_df=2, max_features=60000,
              max_iter=4000, tol=1e-5)
ARABIC_MARKS = re.compile('[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]')
ALIF = str.maketrans({'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا'})


def normalize_arabic(text):
    if not isinstance(text, str):
        raise TypeError('Hemistich must be a string')
    text = unicodedata.normalize('NFKC', text)
    text = ARABIC_MARKS.sub('', text).replace('ـ', '').translate(ALIF)
    text = ''.join(c for c in text if unicodedata.category(c) not in {'Mn', 'Cf'})
    return re.sub(r'\s+', ' ', text).strip()


def normalized_verses(poem, remove_rhyme=True):
    verses = []
    for verse in poem['verses']:
        parts = [normalize_arabic(h) for h in verse]
        if remove_rhyme:
            for j in range(len(parts) - 1, -1, -1):
                if parts[j]:
                    parts[j] = ' '.join(parts[j].split()[:-1])
                    break
        verses.append('|'.join(parts))
    return verses


def poem_to_sequence(poem, remove_rhyme=True):
    return '#'.join(normalized_verses(poem, remove_rhyme))


def load_corpus(data_dir):
    poems = []
    for path in sorted(Path(data_dir).glob('*.json')):
        if path.stem in {'metrics', 'metadata', 'prediction'}:
            continue
        raw = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(raw, list) or not raw or not isinstance(raw[0], dict) or 'verses' not in raw[0]:
            continue
        for p in raw:
            if not isinstance(p.get('verses'), list) or not p['verses']:
                raise ValueError(f'Empty or invalid poem in {path.name}')
            if any(not isinstance(v, list) or len(v) != 2 for v in p['verses']):
                raise ValueError(f'Expected two hemistichs per beyt in {path.name}')
            a = path.stem
            poems.append(dict(p, author=a, source_file=path.name,
                key=f"{a}:{p['id']}", signature=poem_to_sequence(p, False)))
    if len({p['author'] for p in poems}) < 2:
        raise ValueError('Provide the corpus JSON files, one file per author')
    if len({p['key'] for p in poems}) != len(poems):
        raise ValueError('Repeated author:id')
    return poems


def clean_corpus(poems):
    parent = list(range(len(poems)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        parent[root(i)] = root(j)
    seen = {}
    for i, p in enumerate(poems):
        for beyt in p['signature'].split('#'):
            if beyt in seen:
                union(i, seen[beyt])
            else:
                seen[beyt] = i
    groups = defaultdict(list)
    for i in range(len(poems)):
        groups[root(i)].append(i)
    retained, audit = [], []
    for members in groups.values():
        mixed = len({poems[i]['author'] for i in members}) > 1
        best = sorted(members, key=lambda i: (-len(poems[i]['verses']), -len(poems[i]['signature']), poems[i]['key']))[0]
        for i in members:
            p = poems[i]
            status = 'cross_author_quarantine' if mixed else ('retained' if i == best else 'overlap_alternative')
            row = dict(key=p['key'], author=p['author'], poem_id=str(p['id']),
                title=p.get('title', ''), url=p.get('url', ''), n_beyts=len(p['verses']),
                is_mualaka=bool(p.get('is_mualaka', False)), status=status,
                component_size=len(members), representative='' if mixed else poems[best]['key'])
            audit.append(row)
        if not mixed:
            retained.append(poems[best])
    retained.sort(key=lambda p: p['key'])
    return retained, pd.DataFrame(audit).sort_values('key').reset_index(drop=True)


def split_corpus(poems, seed=SEED):
    regular = [p for p in poems if not p.get('is_mualaka', False)]
    mualaka = [p for p in poems if p.get('is_mualaka', False)]
    idx = np.arange(len(regular))
    y = np.array([p['author'] for p in regular])
    dev, test = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    rest, conf = train_test_split(dev, test_size=0.25, random_state=seed+1, stratify=y[dev])
    fit, prob = train_test_split(rest, test_size=0.25, random_state=seed+2, stratify=y[rest])
    splits = {name: [regular[i] for i in ids] for name, ids in
              [('fit', fit), ('probability_calibration', prob), ('conformal_calibration', conf), ('test', test)]}
    splits['mualaka_diagnostic'] = mualaka
    labels = set(y)
    for name, subset in splits.items():
        if name != 'mualaka_diagnostic' and {p['author'] for p in subset} != labels:
            raise ValueError(f'Split {name} is missing authors')
    return splits


def make_training(poems, k=10, remove_rhyme=True):
    texts, labels, owners = [], [], []
    for i, p in enumerate(poems):
        vv = normalized_verses(p, remove_rhyme)
        for start in range(0, len(vv), k):
            texts.append('#'.join(vv[start:start+k]))
            labels.append(p['author'])
            owners.append(i)
    return texts, np.array(labels), np.array(owners)


def make_classifier():
    return LogisticRegression(C=CONFIG['C'], class_weight='balanced',
        solver='lbfgs', max_iter=CONFIG['max_iter'], tol=CONFIG['tol'], random_state=SEED)


def temperature_fit(scores, y, classes):
    lookup = {c: i for i, c in enumerate(classes)}
    indices = np.array([lookup[a] for a in y])
    def loss(log_t):
        q = softmax(scores / np.exp(log_t), axis=1)
        return -np.mean(np.log(np.maximum(q[np.arange(len(y)), indices], 1e-15)))
    opt = minimize_scalar(loss, bounds=(-6, 6), method='bounded')
    if not opt.success:
        raise RuntimeError('Temperature calibration failed')
    return float(np.exp(opt.x))


def conformal_pvalues(probabilities, classes, calibration_by_class):
    result = np.zeros_like(probabilities)
    for j, author in enumerate(classes):
        ref = np.asarray(calibration_by_class[author])
        if not len(ref):
            raise ValueError(f'No conformal reference for {author}')
        scores = 1 - probabilities[:, j]
        result[:, j] = (1 + (ref[:, None] >= scores[None, :]).sum(axis=0)) / (len(ref) + 1)
    return result


def pooled_pvalues(probabilities, calibration_scores):
    ref = np.asarray(calibration_scores)
    return (1 + (ref[:, None, None] >= (1 - probabilities)[None, :, :]).sum(axis=0)) / (len(ref) + 1)


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    out = np.empty(len(p))
    out[order] = np.minimum(1, np.maximum.accumulate((len(p)-np.arange(len(p))) * p[order]))
    return out


def ece(prob, y_idx, bins=10):
    confidence = prob.max(axis=1)
    correct = prob.argmax(axis=1) == y_idx
    edges = np.linspace(0, 1, bins+1)
    return float(sum(np.mean(mask) * abs(correct[mask].mean()-confidence[mask].mean())
        for i in range(bins) if (mask := ((confidence > edges[i]) & (confidence <= edges[i+1]))).any()))


def probability_metrics(prob, y, classes):
    lookup = {a: i for i, a in enumerate(classes)}
    yi = np.array([lookup[a] for a in y])
    onehot = np.eye(len(classes))[yi]
    return dict(log_loss=float(log_loss(y, prob, labels=classes)),
        brier_multiclass=float(np.mean(np.sum((prob-onehot)**2, axis=1))),
        ece_10_bins=ece(prob, yi))


def basic_metrics(y, pred):
    return dict(n_poems=len(y), correct=int(np.sum(np.array(y)==np.array(pred))),
        accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, average='macro', zero_division=0)))


def predictions_frame(poems, model, bundle):
    classes = bundle['classes']
    texts = [poem_to_sequence(p, CONFIG['remove_rhyme']) for p in poems]
    X = bundle['vectorizer'].transform(texts)
    scores = model.decision_function(X)
    q = softmax(scores/bundle['temperature'], axis=1)
    pv = conformal_pvalues(q, classes, bundle['conformal_by_class'])
    pp = pooled_pvalues(q, bundle['conformal_pooled'])
    rows = []
    for i, p in enumerate(poems):
        j = classes.index(p['author'])
        top = int(q[i].argmax())
        row = dict(key=p['key'], author=p['author'], poem_id=str(p['id']), title=p.get('title', ''),
            url=p.get('url', ''), n_beyts=len(p['verses']), predicted_author=classes[top],
            correct=classes[top]==p['author'], probability_claimed=float(q[i,j]),
            probability_top=float(q[i,top]), p_claimed=float(pv[i,j]),
            p_claimed_pooled=float(pp[i,j]), conformal_n_claimed=len(bundle['conformal_by_class'][p['author']]),
            conformal_min_p_claimed=1/(len(bundle['conformal_by_class'][p['author']])+1),
            rejected_at_005=bool(pv[i,j] <= .05),
            set_005=';'.join(a for a, v in zip(classes,pv[i]) if v>.05),
            set_size_005=int((pv[i]>.05).sum()),
            set_010=';'.join(a for a, v in zip(classes,pv[i]) if v>.10),
            set_size_010=int((pv[i]>.10).sum()), zero_features=bool(X[i].nnz==0))
        for k, a in enumerate(classes):
            row[f'prob__{a}'] = float(q[i,k])
            row[f'p__{a}'] = float(pv[i,k])
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame['p_claimed_holm'] = holm_adjust(frame.p_claimed.values)
    frame['rejected_holm_005'] = frame.p_claimed_holm <= .05
    return frame, q, softmax(scores, axis=1)


def bootstrap_metrics(y, pred, classes, n_boot=5000, seed=SEED+10):
    rng = np.random.default_rng(seed)
    y = np.array(y); pred = np.array(pred)
    strata = [np.flatnonzero(y==a) for a in classes]
    out = np.zeros((n_boot, 3))
    for b in range(n_boot):
        ix = np.concatenate([rng.choice(ids, size=len(ids), replace=True) for ids in strata])
        z, h = y[ix], pred[ix]
        hits = z==h
        recalls = [hits[z==a].mean() for a in classes]
        fs = []
        for a in classes:
            tp = np.sum((z==a)&(h==a)); denom = np.sum(z==a)+np.sum(h==a)
            fs.append(2*tp/denom if denom else 0)
        out[b] = hits.mean(), np.mean(recalls), np.mean(fs)
    frame = pd.DataFrame(out, columns=['accuracy','balanced_accuracy','macro_f1'])
    intervals = {col: [float(x) for x in np.quantile(frame[col], [.025,.975])] for col in frame}
    return frame, intervals


def permutation_refit(Xtrain, owners, fit_labels, Xtest, test_labels,
                      observed, classes, n_permutations=199, seed=SEED+20):
    rng = np.random.default_rng(seed)
    fit_labels = np.asarray(fit_labels); test_labels = np.asarray(test_labels)
    scores = []
    for b in range(n_permutations):
        perm_fit = rng.permutation(fit_labels)
        perm_test = rng.permutation(test_labels)
        model = make_classifier().fit(Xtrain, perm_fit[owners])
        pred = model.predict(Xtest)
        value = np.mean([np.mean(pred[perm_test==a]==a) for a in classes])
        scores.append(float(value))
        if (b+1) % 25 == 0:
            print(f'Permutation refits: {b+1}/{n_permutations}', flush=True)
    values = np.array(scores)
    exceeds = int(np.sum(values >= observed-1e-12))
    result = dict(n_permutations=n_permutations, n_scores_ge_observed=exceeds,
        observed_balanced_accuracy=float(observed), p_value=float((1+exceeds)/(1+n_permutations)),
        null_mean=float(values.mean()), null_std=float(values.std(ddof=1)),
        null_max=float(values.max()),
        null_hypothesis='Poem-level author labels exchangeable within fixed fit/test roles, conditional on label counts',
        refit=True, seed=seed)
    return pd.DataFrame({'iteration':np.arange(1,n_permutations+1),'balanced_accuracy':values}), result


def train(data_dir, out_dir, n_permutations=199, n_boot=5000):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    poems = load_corpus(data_dir)
    retained, audit = clean_corpus(poems)
    splits = split_corpus(retained)
    classes = sorted({p['author'] for p in retained})
    roles = {p['key']:role for role, ps in splits.items() for p in ps}
    audit['role'] = audit.key.map(roles).fillna('excluded')
    audit.to_csv(out/'corpus_audit.csv', index=False)
    manifest = audit[audit.role!='excluded'].copy()
    manifest.to_csv(out/'split_manifest.csv', index=False)
    fragments, yf, owners = make_training(splits['fit'])
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=CONFIG['ngram_range'],
        sublinear_tf=True, min_df=CONFIG['min_df'], max_features=CONFIG['max_features'],
        lowercase=False, norm='l2', smooth_idf=True)
    Xtrain = vectorizer.fit_transform(fragments)
    model = make_classifier().fit(Xtrain, yf)
    if model.n_iter_.max() >= CONFIG['max_iter']:
        raise RuntimeError('Classifier did not converge')
    assert list(model.classes_) == classes
    text = lambda role: [poem_to_sequence(p, True) for p in splits[role]]
    labels = lambda role: np.array([p['author'] for p in splits[role]])
    Xprob = vectorizer.transform(text('probability_calibration'))
    temperature = temperature_fit(model.decision_function(Xprob), labels('probability_calibration'), classes)
    conf_q = softmax(model.decision_function(vectorizer.transform(text('conformal_calibration')))/temperature, axis=1)
    lookup = {a:i for i,a in enumerate(classes)}
    conf_y = labels('conformal_calibration')
    conf_scores = 1-conf_q[np.arange(len(conf_y)), [lookup[a] for a in conf_y]]
    by_class = {a: conf_scores[conf_y==a].tolist() for a in classes}
    bundle = dict(schema_version=1, classes=classes, vectorizer=vectorizer, classifier=model,
        temperature=temperature, conformal_by_class=by_class, conformal_pooled=conf_scores.tolist(),
        config=CONFIG, seed=SEED, training_keys=[p['key'] for p in splits['fit']],
        all_reference_keys=[p['key'] for name in ['fit','probability_calibration','conformal_calibration'] for p in splits[name]],
        all_reference_beyts=list({v for name in ['fit','probability_calibration','conformal_calibration']
            for p in splits[name] for v in p['signature'].split('#')}))
    joblib.dump(bundle, out/'m2_model.joblib', compress=3)
    test_frame, tq, rawq = predictions_frame(splits['test'], model, bundle)
    test_frame.to_csv(out/'test_predictions.csv', index=False)
    mualaka_frame, mq, _ = predictions_frame(splits['mualaka_diagnostic'], model, bundle)
    mualaka_frame.to_csv(out/'mualaka_diagnostic.csv', index=False)
    yt = labels('test'); pred = test_frame.predicted_author.values
    metrics = basic_metrics(yt, pred)
    metrics['probability_raw'] = probability_metrics(rawq, yt, classes)
    metrics['probability_temperature'] = probability_metrics(tq, yt, classes)
    bootstrap, intervals = bootstrap_metrics(yt, pred, classes, n_boot)
    bootstrap.to_csv(out/'bootstrap.csv', index=False)
    metrics['bootstrap_95_ci_fixed_model'] = intervals
    Xtest = vectorizer.transform(text('test'))
    permutations, perm_summary = permutation_refit(Xtrain, owners, labels('fit'), Xtest, yt,
        metrics['balanced_accuracy'], classes, n_permutations)
    permutations.to_csv(out/'permutations.csv', index=False)
    metrics['permutation_test'] = perm_summary
    metrics['conformal'] = dict(n_calibration=len(conf_y),
        counts_by_author={a: len(s) for a,s in by_class.items()},
        coverage_005=float((test_frame.p_claimed>.05).mean()),
        mean_set_size_005=float(test_frame.set_size_005.mean()),
        singleton_rate_005=float((test_frame.set_size_005==1).mean()),
        coverage_010=float((test_frame.p_claimed>.1).mean()),
        mean_set_size_010=float(test_frame.set_size_010.mean()),
        pooled_coverage_005=float((test_frame.p_claimed_pooled>.05).mean()),
        rejected_claims_005=int(test_frame.rejected_at_005.sum()),
        rejected_holm_005=int(test_frame.rejected_holm_005.sum()))
    metrics['mualaka_diagnostic'] = basic_metrics(labels('mualaka_diagnostic'), mualaka_frame.predicted_author.values)
    metrics['baseline_most_frequent'] = basic_metrics(yt, np.repeat(Counter(labels('fit')).most_common(1)[0][0],len(yt)))
    metadata = dict(seed=SEED, config=CONFIG, input_poems=len(poems),
        input_beyts=sum(len(p['verses']) for p in poems),
        exact_unique_signatures=len({p['signature'] for p in poems}),
        retained_poems=len(retained), status_counts=audit.status.value_counts().to_dict(),
        split_sizes={k:len(v) for k,v in splits.items()},
        n_training_fragments=len(fragments), n_features=len(vectorizer.vocabulary_),
        optimizer_iterations=int(model.n_iter_.max()), temperature=temperature,
        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(data_dir).glob('*.json'))
            if p.stem in classes})
    (out/'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2),encoding='utf-8')
    (out/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2),encoding='utf-8')
    report = pd.DataFrame(classification_report(yt, pred, labels=classes, output_dict=True,zero_division=0)).T
    report.to_csv(out/'classification_report.csv', index_label='label')
    pd.DataFrame(confusion_matrix(yt, pred,labels=classes),index=classes,columns=classes).to_csv(out/'confusion_matrix.csv',index_label='true_author')
    counts = pd.crosstab(manifest.author, manifest.role).reindex(classes,fill_value=0)
    counts.to_csv(out/'author_split_counts.csv', index_label='author')
    print(json.dumps(dict(metadata=metadata, metrics=metrics),ensure_ascii=False,indent=2),flush=True)
    return bundle, metrics, metadata


def predict(model_path, input_path, output_path):
    bundle = joblib.load(model_path)
    raw = json.loads(Path(input_path).read_text(encoding='utf-8'))
    if isinstance(raw, list):
        if len(raw)!=1:
            raise ValueError('For inference provide exactly one poem object, or a list of length 1')
        raw = raw[0]
    poem = dict(raw)
    claimed = poem.get('claimed_author', poem.get('poet_slug'))
    if claimed not in bundle['classes']:
        raise ValueError('claimed_author must be one of: '+', '.join(bundle['classes']))
    query_beyts = set(normalized_verses(poem, False))
    overlaps = query_beyts & set(bundle['all_reference_beyts'])
    if overlaps:
        raise ValueError('Query shares a normalized beyt with model/calibration reference. '
                         'Rebuild the split excluding its whole overlap component; no valid external p-value is returned.')
    sequence = poem_to_sequence(poem, True)
    X = bundle['vectorizer'].transform([sequence])
    if not re.search(r'[\u0621-\u063a\u0641-\u064a]', sequence) or X.nnz == 0:
        raise ValueError('No usable Arabic text/features after preprocessing')
    scores = bundle['classifier'].decision_function(X)
    q = softmax(scores/bundle['temperature'],axis=1)
    pv = conformal_pvalues(q,bundle['classes'],bundle['conformal_by_class'])[0]
    j = bundle['classes'].index(claimed)
    result = dict(claimed_author=claimed, predicted_author=bundle['classes'][int(q[0].argmax())],
        probabilities=dict(zip(bundle['classes'],q[0].tolist())),
        p_values=dict(zip(bundle['classes'],pv.tolist())), p_claimed=float(pv[j]),
        reject_claimed_at_005=bool(pv[j]<=.05),
        candidate_set_005=[a for a,v in zip(bundle['classes'],pv) if v>.05],
        warning='Closed-set corpus-label probabilities, not historical authenticity; p is not P(author|text). '
                'Known-overlap screening does not rule out paraphrased or unseen dependencies.')
    Path(output_path).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    t = sub.add_parser('train'); t.add_argument('--data-dir',required=True); t.add_argument('--out-dir',required=True)
    t.add_argument('--permutations',type=int,default=199);t.add_argument('--bootstrap',type=int,default=5000)
    p = sub.add_parser('predict');p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.command=='train':
            if args.permutations<1 or args.bootstrap<1: parser.error('Counts must be positive')
            train(args.data_dir,args.out_dir,args.permutations,args.bootstrap)
        else:
            predict(args.model,args.input,args.output)


if __name__=='__main__':
    main()
