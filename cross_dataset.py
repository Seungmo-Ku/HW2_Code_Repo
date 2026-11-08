"""
교차 데이터셋 실험: 우리 데이터(AI Hub 공학 단락) <-> KatFish 초록 (Park et al., 2025)
분포가 바뀌어도 n-gram 탐지기와 KatFishNet 쉼표 특징이 버티는지, 단위별 최적 n 이 유지되는지 확인

방향 (seed 마다 두 데이터의 분할을 새로 뽑음):
  ours->ours       : 우리 학습 분할(80%, 생성기 하나)로 학습 -> 우리 테스트 분할, 학습에 안 쓴 생성기 2개 평균 (experiment.py 의 unseen 과 같음)
  ours->katfish    : 같은 모델 -> KatFish 초록 전체 (사람 100, LLM 278), 학습 생성기 3개 평균
  katfish->katfish : KatFish 초록 80% (라벨 층화)로 학습 -> 나머지 20%
  katfish->ours    : 같은 모델 -> 우리 테스트 분할, 생성기 3개 평균
KatFish 쪽은 길이 맞춤을 하지 않음 (사람/LLM 짝이 없음)

사용법: python cross_dataset.py [--mask] [--units ...] [--ns ...]
"""
import argparse, csv, json, os, random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from experiment import load, normalize, mask_format, ngrams, TOKENIZERS, HERE
from katfishnet_baseline import comma_features

DIRECTIONS = ["ours->ours", "ours->katfish", "katfish->katfish", "katfish->ours"]


def load_katfish(path, prep):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return [prep(r["text"]) for r in rows], [0 if r["written_by"] == "human" else 1 for r in rows]

def split_katfish(labels, seed):
    # 라벨별로 섞어 80/20 (사람 100 : LLM 278 비율 유지)
    rng = random.Random(seed)
    tr, te = [], []
    for y in (0, 1):
        idx = [i for i, l in enumerate(labels) if l == y]
        rng.shuffle(idx)
        cut = int(len(idx) * 0.8)
        tr += idx[:cut]; te += idx[cut:]
    return tr, te

def split_ours(ids, seed):
    rng = random.Random(seed)
    shuffled = ids[:]; rng.shuffle(shuffled)
    cut = int(len(shuffled) * 0.8)
    return shuffled[:cut], shuffled[cut:]


def run(fit, score, ids, pairs, kf_labels, seed):
    """fit(학습 텍스트 번호들, 라벨) -> 모델, score(모델, 텍스트 번호들) -> 점수. 방향별 AUROC 를 돌려줌"""
    gens = list(pairs)
    tr_ids, te_ids = split_ours(ids, seed)
    kf_tr, kf_te = split_katfish(kf_labels, seed)
    ours = lambda g, ids_: [(g, i, k) for i in ids_ for k in (0, 1)]
    y_ours = lambda ids_: [k for _ in ids_ for k in (0, 1)]
    kf = lambda idx: [("kf", j) for j in idx]
    kf_all = list(range(len(kf_labels)))

    res = {d: [] for d in DIRECTIONS}
    for g_tr in gens:
        m = fit(ours(g_tr, tr_ids), y_ours(tr_ids))
        for g_te in gens:
            if g_te != g_tr:
                res["ours->ours"].append(roc_auc_score(y_ours(te_ids), score(m, ours(g_te, te_ids))))
        res["ours->katfish"].append(roc_auc_score(kf_labels, score(m, kf(kf_all))))
    m = fit(kf(kf_tr), [kf_labels[j] for j in kf_tr])
    res["katfish->katfish"].append(roc_auc_score([kf_labels[j] for j in kf_te], score(m, kf(kf_te))))
    for g_te in gens:
        res["katfish->ours"].append(roc_auc_score(y_ours(te_ids), score(m, ours(g_te, te_ids))))
    return {d: np.mean(v) for d, v in res.items()}


def main(args):
    prep = mask_format if args.mask else normalize
    ids, pairs = load(args)
    kf_texts, kf_labels = load_katfish(args.katfish, prep)
    print(f"우리 문서 {len(ids)}개 x 생성기 {len(pairs)}개 / KatFish 초록 사람 {kf_labels.count(0)} LLM {kf_labels.count(1)}"
          f"{' (mask)' if args.mask else ''}")

    # 텍스트 키: 우리 = (생성기, 문서 id, 0 사람/1 LLM), KatFish = ("kf", 행 번호)
    texts = {(g, i, k): pairs[g][i][k] for g in pairs for i in ids for k in (0, 1)}
    texts.update({("kf", j): t for j, t in enumerate(kf_texts)})
    keys = list(texts); pos = {key: p for p, key in enumerate(keys)}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = open(args.out, "w", newline="", encoding="utf-8")
    w = csv.writer(out)
    w.writerow(["method", "n", "seed", "direction", "auroc"])
    summary = {}

    def record(method, n, per_seed):
        for seed, res in zip(args.seeds, per_seed):
            for d in DIRECTIONS:
                w.writerow([method, n, seed, d, f"{res[d]:.4f}"])
        summary[(method, n)] = {d: (np.mean([r[d] for r in per_seed]), np.std([r[d] for r in per_seed])) for d in DIRECTIONS}
        print(f"{method:9s} n={n}  " + "  ".join(f"{d}={summary[(method, n)][d][0] * 100:5.1f}" for d in DIRECTIONS), flush=True)
        out.flush()

    # KatFishNet 쉼표 특징 5개
    feats = {key: comma_features(texts[key]) for key in keys}
    def fit_comma(tr, y):
        X = [feats[key] for key in tr]
        scaler = StandardScaler().fit(X)
        return scaler, LogisticRegression(random_state=0).fit(scaler.transform(X), y)
    def score_comma(m, te):
        scaler, clf = m
        return clf.predict_proba(scaler.transform([feats[key] for key in te]))[:, 1]
    record("comma", 0, [run(fit_comma, score_comma, ids, pairs, kf_labels, s) for s in args.seeds])

    # n-gram TF-IDF + LR (experiment.py 와 같은 설정)
    for unit in args.units:
        toks = [TOKENIZERS[unit](texts[key]) for key in keys]
        for n in args.ns:
            def fit_ngram(tr, y):
                vec = TfidfVectorizer(analyzer=lambda p: ngrams(toks[p], n), max_features=args.max_features, min_df=2)
                clf = LogisticRegression(max_iter=2000, random_state=0).fit(vec.fit_transform([pos[key] for key in tr]), y)
                return vec, clf
            def score_ngram(m, te):
                vec, clf = m
                return clf.predict_proba(vec.transform([pos[key] for key in te]))[:, 1]
            record(unit, n, [run(fit_ngram, score_ngram, ids, pairs, kf_labels, s) for s in args.seeds])
    out.close()

    print("\n방향별 최고 성능과 최적 n (최고값의 1 표준편차 이내인 가장 작은 n):")
    for d in DIRECTIONS:
        line = f"  {d:17s} comma {summary[('comma', 0)][d][0] * 100:5.1f}"
        for unit in args.units:
            rows = [(n, *summary[(unit, n)][d]) for n in args.ns]
            best_n, best, best_std = max(rows, key=lambda r: r[1])
            opt = min(n for n, m, _ in rows if m >= best - best_std)
            line += f" | {unit} 최고 {best * 100:5.1f} (n={best_n}) 최적 n={opt}"
        print(line)
    print(f"seed/방향별 결과 -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--human", default=os.path.join(HERE, "data", "sampled.jsonl"))
    p.add_argument("--gen_dir", default=os.path.join(HERE, "data"))
    p.add_argument("--katfish", default=os.path.join(HERE, "..", "HW1_Code", "katfishnet", "katfish_dataset", "abstract.jsonl"))
    p.add_argument("--units", nargs="+", default=["eojeol", "morpheme", "syllable"])
    p.add_argument("--ns", nargs="+", type=int, default=list(range(1, 9)))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--max_features", type=int, default=10000)
    p.add_argument("--mask", action="store_true")
    p.add_argument("--out", default=os.path.join(HERE, "results", "cross.csv"))
    main(p.parse_args())
