"""
§5.1 근거: 어절 5-gram 특징이 테스트 글에 얼마나 나타나는지 (마스킹 데이터)
experiment.py 와 같은 분할/특징 선택(min_df=2, 최대 10,000개)으로, 학습 문서 수별로
'선택된 어절 5-gram 을 하나라도 포함한 unseen 테스트 글의 비율' 을 seed 별로 출력한다.

사용법: python sparsity.py [--n 5] [--sizes 25 200 797]
"""
import argparse, os, random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from experiment import load, ngrams, TOKENIZERS, HERE


def main(args):
    ids, pairs = load(args)
    gens = list(pairs)
    for seed in args.seeds:
        rng = random.Random(seed)
        shuffled = ids[:]; rng.shuffle(shuffled)
        cut_at = int(len(shuffled) * 0.8)
        tr_ids, te_ids = shuffled[:cut_at], shuffled[cut_at:]
        out = []
        for size in args.sizes:
            fr, nf = [], []
            for g_tr in gens:
                vec = TfidfVectorizer(analyzer=lambda t: ngrams(TOKENIZERS["eojeol"](t), args.n), max_features=10000, min_df=2)
                vec.fit([pairs[g_tr][i][k] for i in tr_ids[:size] for k in (0, 1)])
                X = vec.transform([pairs[g_te][i][k] for g_te in gens if g_te != g_tr for i in te_ids for k in (0, 1)])
                fr.append(np.mean(X.getnnz(axis=1) > 0)); nf.append(len(vec.vocabulary_))
            out.append(f"{size}문서: 특징 {np.mean(nf):.0f}개, 포함 글 {np.mean(fr) * 100:.1f}%")
        print(f"seed {seed} | 어절 {args.n}-gram | " + " / ".join(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--human", default=os.path.join(HERE, "data", "sampled.jsonl"))
    p.add_argument("--gen_dir", default=os.path.join(HERE, "data"))
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--sizes", nargs="+", type=int, default=[25, 200, 797])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--no_mask", dest="mask", action="store_false", help="기본은 마스킹 데이터 (§5.1 과 같음)")
    main(p.parse_args())
