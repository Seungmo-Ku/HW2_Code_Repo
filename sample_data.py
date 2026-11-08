"""
AI Hub 논문요약 데이터에서 사람 글(논문 본문 단락)을 샘플링한다.
  - 분야: 공학 (ipc == '공학')
  - 연도: 1992~2020 (LLM 상용화 이전)
  - 길이: 너무 짧거나 긴 단락 제외 (--min_len ~ --max_len 자)
각 문서의 summary_section[0] 을 사용:
  orginal_text -> 사람 글 (human)
  summary_text -> LLM 생성 시 입력으로 주는 요약
사용법: python sample_data.py --n 1000 --out data/sampled.jsonl
"""
import argparse, glob, json, os, random

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           "018.논문자료 요약 데이터", "01.데이터", "1. Training",
                           "1. 라벨링데이터_231101_add", "training_논문")


def load_candidates(data_dir, args):
    cands = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            docs = json.load(f)[0]["data"]
        for d in docs:
            if d.get("ipc") != args.field:
                continue
            year = d.get("date", "")[:4]
            if not year.isdigit() or not (args.min_year <= int(year) <= args.max_year):
                continue
            sec = (d.get("summary_section") or [{}])[0]
            text = (sec.get("orginal_text") or "").strip()
            summary = (sec.get("summary_text") or "").strip()
            if not (args.min_len <= len(text) <= args.max_len) or not summary:
                continue
            cands.append({"id": d["doc_id"], "title": d["title"], "year": int(year),
                          "summary": summary, "human_text": text})
        del docs  # 파일이 커서 하나씩 읽고 바로 해제
    return cands


def main(args):
    cands = load_candidates(args.data_dir, args)
    print(f"조건을 만족하는 문서: {len(cands)}개")
    random.Random(args.seed).shuffle(cands)
    sample = cands[:args.n]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lens = sorted(len(r["human_text"]) for r in sample)
    print(f"{len(sample)}개 저장 -> {args.out} (길이 중앙값 {lens[len(lens) // 2]}자)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=DEFAULT_DIR)
    p.add_argument("--out", default="data/sampled.jsonl")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--field", default="공학")
    p.add_argument("--min_year", type=int, default=1992)
    p.add_argument("--max_year", type=int, default=2020)
    p.add_argument("--min_len", type=int, default=300)
    p.add_argument("--max_len", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
