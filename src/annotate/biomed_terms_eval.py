#!/usr/bin/env python3
"""

Usage (examples):
    python biomed_terms_eval.py \
      --dataset_name sethjsa/tico_en_fr \
      --src_field en \
      --tgt_field fr \
      --src_lang en \
      --tgt_lang fr \
      --split "test[:10]"

  # With predictions:
  python biomed_terms_eval.py ... --predictions "preds.jsonl" --out_eval "eval_terms.jsonl"

Env:
  export DEEPSEEK_API_KEY=sk-...

Or whatever the equivalent is with conda
"""
import os, re, json, argparse, time, asyncio, logging, sys
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI, AsyncOpenAI
import unicodedata

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("biomed_terms_eval")

# deepseek client (openai-compatible)
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# flags (set via CLI)
DEBUG_REJECTIONS = False
RAW_MODEL_LOG = None
DIAGNOSTIC_TOLERANT = False

# tokenization utils
_token_pat = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def tokens_with_offsets(s):
    return [(m.group(0), m.start(), m.end()) for m in _token_pat.finditer(s)]

def span_to_token_span(tokens, cstart, cend):
    touched = [i for i,(_,ts,te) in enumerate(tokens) if not (te <= cstart or ts >= cend)]
    if not touched:
        dists = [(i, min(abs(ts-cstart), abs(te-cend))) for i,(_,ts,te) in enumerate(tokens)]
        if not dists: return (0,0)
        i = min(dists, key=lambda x:x[1])[0]
        return (i, i+1)
    return (touched[0], touched[-1]+1)

def norm_space(s):
    return re.sub(r"\s+", " ", s.strip())

def normalize_for_match(s):
    s = s.lower()
    s = re.sub(r"[“”\"'`´’]", "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*-\s*", "-", s)
    return s.strip()

# regex builders (robust)
def _needle_to_regex(needle: str) -> re.Pattern:
    n = unicodedata.normalize("NFC", needle.strip())
    pat = re.escape(n)
    pat = pat.replace(r"\ ", r"\s+")
    pat = pat.replace(r"\-", r"[-\u2011\u2013\u2014]?")
    pat = pat.replace(r"\"", r"[\"“”]?").replace(r"\'", r"[\'’´`]?")
    return re.compile(pat, flags=re.IGNORECASE)

def tolerant_regex(needle: str) -> re.Pattern:
    n = unicodedata.normalize("NFKD", needle).replace("ß", "ss")
    n = "".join(ch for ch in n if not unicodedata.combining(ch))
    pat = re.escape(n)
    pat = pat.replace(r"\ ", r"\s+")
    pat = pat.replace(r"\-", r"[-\u2011\u2013\u2014]?")
    return re.compile(pat, flags=re.IGNORECASE)


# find occurrences (with debug)
def find_occurrence_debug(hay, needle, used):
    """
    Try to find a non-overlapping occurrence of `needle` in `hay`.
    Returns (span, error_code, regex_pattern, diagnostic_flag).
      span: (start, end) or None
      error_code: None | "empty_needle" | "no_match" | "overlap"
      regex_pattern: the concrete regex used
      diagnostic_flag: None | "no_match_but_tolerant_hit"
    """
    if not isinstance(needle, str) or not needle.strip():
        return None, "empty_needle", None, None

    pat = _needle_to_regex(needle)
    for m in pat.finditer(hay):
        span = (m.start(), m.end())
        if all(span[1] <= u[0] or span[0] >= u[1] for u in used):
            return span, None, pat.pattern, None

    # optional tolerant ping
    diag = None
    if DIAGNOSTIC_TOLERANT:
        H = unicodedata.normalize("NFKD", hay).replace("ß", "ss")
        H = "".join(ch for ch in H if not unicodedata.combining(ch))
        if tolerant_regex(needle).search(H):
            diag = "no_match_but_tolerant_hit"

    return None, "no_match", pat.pattern, diag

# model prompts
SYS_PROMPT = (
"You are a precise biomedical term aligner. Given a source and target sentence, "
"identify (bio)medical or scientific, domain-specific terms (single or multiword: diseases, syndromes, procedures, drugs, genes, "
"pathologies, anatomy) that appear in the source, and return their exact translations that appear "
"in the target. Only include pairs if both surface forms appear verbatim in their respective sentences. "
"Output strictly JSON with a 'pairs' array of objects: "
"{\"pairs\":[{\"src\":\"...\",\"tgt\":\"...\"}, ...]} and nothing else."
)

# json helpers
def _parse_pairs(text: str):
    # quick JSON parse with a couple of fallbacks
    try:
        data = json.loads(text)
    except Exception:
        t = text.strip()
        if t.startswith("```"):
            t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
        if "{" in t and "}" in t:
            t = t[t.find("{"): t.rfind("}") + 1]
        try:
            data = json.loads(t)
        except Exception:
            return []

    pairs = data.get("pairs", []) if isinstance(data, dict) else []
    out = []
    for p in pairs:
        src_t = norm_space(str(p.get("src", "")))
        tgt_t = norm_space(str(p.get("tgt", "")))
        if src_t and tgt_t:
            out.append({"src": src_t, "tgt": tgt_t})
    return out


def append_jsonl(path, rec):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning(f"Failed to append to {path}: {e!r}")

# sync model call
def deepseek_extract_pairs_sync(client, src, tgt, src_lang, tgt_lang, retries=3, sleep_s=2.0):
    user_prompt = (
        f"Source language: {src_lang}\n"
        f"Target language: {tgt_lang}\n\n"
        f"Source sentence:\n{src}\n\n"
        f"Target sentence:\n{tgt}\n\n"
        "Return JSON now."
    )
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role":"system","content":SYS_PROMPT},
                          {"role":"user","content":user_prompt}],
                stream=False,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content.strip()
            pairs = _parse_pairs(text)

            if RAW_MODEL_LOG:
                append_jsonl(RAW_MODEL_LOG, {
                    "mode":"sync", "text":text, "parsed_pairs": pairs,
                    "src": src, "tgt": tgt
                })

            return pairs
        except Exception as e:
            if attempt == retries - 1:
                log.warning(f"DeepSeek sync call failed after retries: {e!r}")
                if RAW_MODEL_LOG:
                    append_jsonl(RAW_MODEL_LOG, {"mode":"sync_error", "error": repr(e), "src": src, "tgt": tgt})
                return []
            time.sleep(sleep_s)
    return []

# async model call (with limiter/retries)
def make_async_caller(async_client, limiter, retries, retry_min, retry_max):
    @retry(
        reraise=True,
        stop=stop_after_attempt(retries),
        wait=wait_exponential(min=retry_min, max=retry_max),
        retry=retry_if_exception_type(Exception),
    )
    async def _call(src, tgt, src_lang, tgt_lang):
        user_prompt = (
            f"Source language: {src_lang}\n"
            f"Target language: {tgt_lang}\n\n"
            f"Source sentence:\n{src}\n\n"
            f"Target sentence:\n{tgt}\n\n"
            "Return JSON now."
        )
        async with limiter:
            resp = await async_client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role":"system","content":SYS_PROMPT},
                          {"role":"user","content":user_prompt}],
                stream=False,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        text = resp.choices[0].message.content.strip()
        pairs = _parse_pairs(text)

        if RAW_MODEL_LOG:
            append_jsonl(RAW_MODEL_LOG, {
                "mode":"async", "text":text, "parsed_pairs": pairs,
                "src": src, "tgt": tgt
            })

        return pairs
    return _call

# build annotations
def build_annotation(idx, src, tgt, pairs):
    src_tokens = tokens_with_offsets(src)
    tgt_tokens = tokens_with_offsets(tgt)
    used_src, used_tgt, term_pairs = [], [], []
    rejected = []

    for p in pairs:
        s_term, t_term = p["src"], p["tgt"]

        s_span, s_err, s_pat, s_diag = find_occurrence_debug(src, s_term, used_src)
        t_span, t_err, t_pat, t_diag = find_occurrence_debug(tgt, t_term, used_tgt)

        # sanity checks
        if not s_span or not t_span:
            if DEBUG_REJECTIONS:
                rejected.append({
                    "src_term": s_term,
                    "tgt_term": t_term,
                    "src_error": s_err,
                    "tgt_error": t_err,
                    "src_regex": s_pat,
                    "tgt_regex": t_pat,
                    "src_diag": s_diag,
                    "tgt_diag": t_diag,
                    "src_preview": src[:200],
                    "tgt_preview": tgt[:200],
                })
            continue

        used_src.append(s_span)
        used_tgt.append(t_span)
        s_ts, s_te = span_to_token_span(src_tokens, *s_span)
        t_ts, t_te = span_to_token_span(tgt_tokens, *t_span)
        term_pairs.append({
            "src": {"text": s_term, "char_start": s_span[0], "char_end": s_span[1], "token_start": s_ts, "token_end": s_te},
            "tgt": {"text": t_term, "char_start": t_span[0], "char_end": t_span[1], "token_start": t_ts, "token_end": t_te},
        })

    ann = {"idx": idx, "src": src, "tgt": tgt, "term_pairs": term_pairs}
    if DEBUG_REJECTIONS:
        ann["debug_rejections"] = rejected
    return ann

# eval helpers
def exact_match_terms(tgt_terms, pred_sent):
    pred_norm = normalize_for_match(pred_sent)
    results, hit = [], 0
    for t in tgt_terms:
        ok = normalize_for_match(t) in pred_norm
        results.append({"term": t, "match": ok})
        if ok: hit += 1
    denom = len(tgt_terms) if tgt_terms else 0
    acc = (hit / denom) if denom else None
    return {"matched": hit, "total": denom, "accuracy": acc, "details": results}

# io helpers
def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

# main pipeline (sync)
def extract_and_annotate(args):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    ds = load_dataset(args.dataset_name, split=args.split)

    def get_field(rec, dotted):
        cur = rec
        for part in dotted.split("."):
            cur = cur[part]
        return cur

    out_recs = []
    it = ds if not args.max_examples else ds.select(range(min(args.max_examples, len(ds))))
    for i, rec in enumerate(it):
        src = get_field(rec, args.src_field)
        tgt = get_field(rec, args.tgt_field)

        # sanity checks
        if not isinstance(src, str) or not isinstance(tgt, str) or not src.strip() or not tgt.strip():
            log.warning(f"idx={i} empty or non-string field: src={type(src)} len={len(str(src))}, tgt={type(tgt)} len={len(str(tgt))}")
            ann = {"idx": i, "src": str(src), "tgt": str(tgt), "term_pairs": []}
            if DEBUG_REJECTIONS:
                ann["debug_rejections"] = [{"src_term":"", "tgt_term":"", "src_error":"empty_src_or_tgt"}]
            out_recs.append(ann)
            continue

        pairs = deepseek_extract_pairs_sync(client, src, tgt, args.src_lang, args.tgt_lang,
                                            retries=args.retries, sleep_s=args.retry_sleep)
        ann = build_annotation(i, src, tgt, pairs)
        out_recs.append(ann)

    if args.out_annotations:
        write_jsonl(args.out_annotations, out_recs)
    return out_recs

# main pipeline (async)
async def extract_and_annotate_async(args):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    aclient = AsyncOpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    ds = load_dataset(args.dataset_name, split=args.split)

    def get_field(rec, dotted):
        cur = rec
        for part in dotted.split("."):
            cur = cur[part]
        return cur

    it = ds if not args.max_examples else ds.select(range(min(args.max_examples, len(ds))))
    limiter = AsyncLimiter(args.rate_limit, time_period=1)
    caller = make_async_caller(
        async_client=aclient,
        limiter=limiter,
        retries=args.retries,
        retry_min=max(0.5, args.retry_sleep/2),
        retry_max=max(2.0, args.retry_sleep*4),
    )

    sem = asyncio.Semaphore(args.concurrency)
    results = [None] * len(it)

    async def worker(i, src, tgt):
        async with sem:
            try:
                # sanity checks
                if not isinstance(src, str) or not isinstance(tgt, str) or not src.strip() or not tgt.strip():
                    log.warning(f"idx={i} empty or non-string field: src={type(src)} len={len(str(src))}, tgt={type(tgt)} len={len(str(tgt))}")
                    rec = {"idx": i, "src": str(src), "tgt": str(tgt), "term_pairs": []}
                    if DEBUG_REJECTIONS:
                        rec["debug_rejections"] = [{"src_term":"", "tgt_term":"", "src_error":"empty_src_or_tgt"}]
                    results[i] = rec
                    return

                pairs = await caller(src, tgt, args.src_lang, args.tgt_lang)
            except Exception as e:
                log.warning(f"idx={i} failed: {e!r}")
                pairs = []
            results[i] = build_annotation(i, src, tgt, pairs)

    tasks = []
    for i, rec in enumerate(it):
        src = get_field(rec, args.src_field)
        tgt = get_field(rec, args.tgt_field)
        tasks.append(worker(i, src, tgt))

    await tqdm_asyncio.gather(*tasks, desc="Extracting term pairs (concurrent)")
    out_recs = [r for r in results if r is not None]

    if args.out_annotations:
        write_jsonl(args.out_annotations, out_recs)
    return out_recs

# evaluation
def evaluate_predictions(args, annotations=None):
    if annotations is None:
        if not args.out_annotations:
            raise RuntimeError("Annotations not provided; specify --out_annotations or pass in-memory data.")
        annotations = read_jsonl(args.out_annotations)
    preds = read_jsonl(args.predictions)

    pred_map = {i: r["prediction"] for i, r in enumerate(preds)}
    eval_lines, total_hit, total_terms = [], 0, 0

    for rec in annotations:
        idx = rec["idx"]
        tgt_terms = [p["tgt"]["text"] for p in rec["term_pairs"]]
        pred = pred_map.get(idx, "")
        em = exact_match_terms(tgt_terms, pred)
        if em["total"] and em["total"] > 0:
            total_hit += em["matched"]
            total_terms += em["total"]
        eval_lines.append({
            "idx": idx,
            "prediction": pred,
            "target_terms": tgt_terms,
            "matched": em["matched"],
            "total": em["total"],
            "accuracy": em["accuracy"],
            "details": em["details"],
        })

    summary = {
        "micro_accuracy": (total_hit/total_terms) if total_terms else None,
        "total_terms": total_terms,
        "total_matched": total_hit,
        "num_examples": len(annotations),
    }
    if args.out_eval:
        write_jsonl(args.out_eval, eval_lines)
        with open(args.out_eval + ".summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return {"summary": summary, "per_example": eval_lines}

# cli
def parse_args():
    p = argparse.ArgumentParser(description="Biomedical term extraction + exact-match evaluation (concurrent)")
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--split", default="train[:100]")
    p.add_argument("--src_field", required=True, help='e.g. "translation.en" or "en"')
    p.add_argument("--tgt_field", required=True, help='e.g. "translation.de" or "de"')
    p.add_argument("--src_lang", default="English")
    p.add_argument("--tgt_lang", default="German")
    p.add_argument("--max_examples", type=int, default=None)

    p.add_argument("--out_annotations", default="annotations.jsonl")
    p.add_argument("--predictions", default=None, help='JSONL with {"prediction": "..."} per line, aligned to dataset order')
    p.add_argument("--out_eval", default="eval_terms.jsonl")

    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry_sleep", type=float, default=2.0)

    # concurrency
    p.add_argument("--concurrency", type=int, default=8, help="Max in-flight requests")
    p.add_argument("--rate_limit", type=int, default=15, help="Requests per second")
    p.add_argument("--no_async", action="store_true", help="Force synchronous mode")

    # debugging
    p.add_argument("--debug_rejections", action="store_true", help="Include per-example rejection reasons in annotations.jsonl")
    p.add_argument("--raw_model_log", default=None, help="Path to append raw model outputs and parsed pairs")
    p.add_argument("--diagnostic_tolerant", action="store_true", help="Run tolerant secondary match to flag near-misses")
    return p.parse_args()

# entrypoint
def main():
    global DEBUG_REJECTIONS, RAW_MODEL_LOG, DIAGNOSTIC_TOLERANT
    args = parse_args()
    DEBUG_REJECTIONS = args.debug_rejections
    RAW_MODEL_LOG = args.raw_model_log
    DIAGNOSTIC_TOLERANT = args.diagnostic_tolerant

    if args.no_async or args.concurrency <= 1:
        annotations = extract_and_annotate(args)
    else:
        annotations = asyncio.run(extract_and_annotate_async(args))

    if args.predictions:
        result = evaluate_predictions(args, annotations)
        print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    else:
        print(f"Wrote {len(annotations)} annotations to {args.out_annotations}")

if __name__ == "__main__":
    main()
