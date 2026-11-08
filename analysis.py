"""
분석: (1) 단위별 최적 n 의 상위 특징, (2) 부트스트랩으로 AUROC 차이의 95% 신뢰구간

(1) 상위 특징: 997문서 x 생성기 3개 전체로 학습한 로지스틱 회귀 계수의 양 끝 (+ = LLM, - = 사람)
(2) 부트스트랩: experiment.py 와 같은 분할/모델로 unseen 테스트 점수를 구한 뒤,
    seed 마다 테스트 문서(사람 글/LLM 글 짝 단위)를 복원 추출해 두 설정의 unseen AUROC 차이를 계산.
    한 번의 반복 = seed 3개 각각 재추출 후 평균. 반복 1000번의 2.5/97.5 백분위를 신뢰구간으로 씀
    비교: 단위 간 (각 최적 n 끼리), 단위 안에서 최적 n vs 바로 옆 n

사용법: python analysis.py [--mask]
"""
import argparse, os, random
import numpy as np
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from experiment import load, ngrams, TOKENIZERS, HERE

OPT = {"eojeol": 1, "morpheme": 2, "syllable": 4}
CONFIGS = [("eojeol", 1), ("eojeol", 2), ("morpheme", 1), ("morpheme", 2), ("morpheme", 3),
           ("syllable", 3), ("syllable", 4), ("syllable", 5)]


def auroc(y, s):
    # 순위 기반 AUROC (동점은 평균 순위). roc_auc_score 와 같은 값이지만 부트스트랩 반복에서 빠름
    ranks = rankdata(s)
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def fit(docs, y, n, max_features):
    vec = TfidfVectorizer(analyzer=lambda d: ngrams(d, n), max_features=max_features, min_df=2)
    clf = LogisticRegression(max_iter=2000, random_state=0).fit(vec.fit_transform(docs), y)
    return vec, clf


def top_features(ids, pairs, toks, k):
    print("\n## 상위 특징 (전체 데이터, + = LLM 쪽, - = 사람 쪽)")
    gens = list(pairs)
    for unit, n in OPT.items():
        docs = [toks[unit][(g, i, j)] for g in gens for i in ids for j in (0, 1)]
        y = [j for g in gens for i in ids for j in (0, 1)]
        vec, clf = fit(docs, y, n, 10000)
        names = vec.get_feature_names_out(); coef = clf.coef_[0]
        order = np.argsort(coef)
        fmt = lambda idx: ", ".join(f"'{names[t]}'({coef[t]:+.1f})" for t in idx)
        print(f"{unit} n={n}\n  LLM : {fmt(order[::-1][:k])}\n  사람: {fmt(order[:k])}")


def bootstrap(ids, pairs, toks, seeds, n_boot):
    gens = list(pairs)
    # scores[config][seed] = (테스트 문서 수, {(학습, 테스트 생성기): 점수 배열 [문서, 사람/LLM]})
    scores = {c: {} for c in CONFIGS}
    for seed in seeds:
        rng = random.Random(seed)
        shuffled = ids[:]; rng.shuffle(shuffled)
        cut_at = int(len(shuffled) * 0.8)
        tr_ids, te_ids = shuffled[:cut_at], shuffled[cut_at:]
        for unit, n in CONFIGS:
            per_pair = {}
            for g_tr in gens:
                vec, clf = fit([toks[unit][(g_tr, i, j)] for i in tr_ids for j in (0, 1)],
                               [j for i in tr_ids for j in (0, 1)], n, 10000)
                for g_te in gens:
                    if g_te != g_tr:
                        s = clf.predict_proba(vec.transform([toks[unit][(g_te, i, j)] for i in te_ids for j in (0, 1)]))[:, 1]
                        per_pair[(g_tr, g_te)] = s.reshape(-1, 2)
            scores[(unit, n)][seed] = per_pair
        print(f"  seed {seed} 점수 계산 완료", flush=True)

    # 반복마다 seed 별로 같은 재추출 인덱스를 모든 설정에 써서 짝지은 차이를 구함
    rs = np.random.RandomState(0)
    boot = {c: np.zeros(n_boot) for c in CONFIGS}
    for b in range(n_boot):
        for seed in seeds:
            n_te = next(iter(scores[CONFIGS[0]][seed].values())).shape[0]
            idx = rs.randint(0, n_te, n_te)
            y = np.tile([0, 1], n_te)
            for c in CONFIGS:
                boot[c][b] += np.mean([auroc(y, s[idx].ravel()) for s in scores[c][seed].values()]) / len(seeds)
    point = {c: np.mean([np.mean([auroc(np.tile([0, 1], s.shape[0]), s.ravel()) for s in scores[c][seed].values()])
                         for seed in seeds]) for c in CONFIGS}

    def report(a, b):
        d = (boot[a] - boot[b]) * 100
        lo, hi = np.percentile(d, [2.5, 97.5])
        sig = "유의" if lo > 0 or hi < 0 else "유의하지 않음"
        print(f"  {a[0]} {a[1]} ({point[a] * 100:.1f}) - {b[0]} {b[1]} ({point[b] * 100:.1f}) = "
              f"{(point[a] - point[b]) * 100:+.2f}%p  95% CI [{lo:+.2f}, {hi:+.2f}]  {sig}")

    print(f"\n## 부트스트랩 {n_boot}회: 단위 간 (각 최적 n)")
    report(("syllable", 4), ("eojeol", 1))
    report(("syllable", 4), ("morpheme", 2))
    report(("eojeol", 1), ("morpheme", 2))
    print("\n## 부트스트랩: 최적 n vs 옆 n")
    report(("eojeol", 1), ("eojeol", 2))
    report(("morpheme", 2), ("morpheme", 1))
    report(("morpheme", 2), ("morpheme", 3))
    report(("syllable", 4), ("syllable", 3))
    report(("syllable", 4), ("syllable", 5))


def main(args):
    ids, pairs = load(args)
    print(f"문서 {len(ids)}개{' (mask)' if args.mask else ''}")
    units = sorted(OPT)
    toks = {u: {(g, i, j): TOKENIZERS[u](pairs[g][i][j]) for g in pairs for i in ids for j in (0, 1)} for u in units}
    top_features(ids, pairs, toks, args.k)
    print("\n## 부트스트랩 준비")
    bootstrap(ids, pairs, toks, args.seeds, args.n_boot)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--human", default=os.path.join(HERE, "data", "sampled.jsonl"))
    p.add_argument("--gen_dir", default=os.path.join(HERE, "data"))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--k", type=int, default=15, help="상위 특징 개수")
    p.add_argument("--mask", action="store_true")
    main(p.parse_args())
