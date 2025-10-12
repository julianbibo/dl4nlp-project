#!/usr/bin/env python3
"""
File-based biomedical term extraction + exact-match evaluation.

Usage:
  python src/annotate/biomed_terms_eval_files.py \
    --corpus_file data/wmt22/en_fr_eval_ids.txt \
    --data_dir data/wmt22/en_fr \
    --src en --tgt fr \
    --tgt_lang French \
    --out_annotations annotations_en_fr.jsonl \
    --concurrency 8 --rate_limit 15

  # With predictions (JSONL: one {"prediction":"..."} per line, aligned to corpus order)
  python biomed_terms_eval_files.py ... --predictions preds.jsonl --out_eval eval_terms.jsonl

Env:
  export DEEPSEEK_API_KEY=sk-...
"""

import os
import re
import json
import argparse
import time
import asyncio
import logging
import sys
import unicodedata
from aiolimiter import AsyncLimiter
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from openai import OpenAI, AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv


load_dotenv("./environment/.env")


# ---------- logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("biomed_terms_eval_files")

# ---------- model ----------
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# flags (set via CLI)
DEBUG_REJECTIONS = False
RAW_MODEL_LOG = None
DIAGNOSTIC_TOLERANT = False

# ---------- tokenization / text utils ----------
_token_pat = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokens_with_offsets(s):
    return [(m.group(0), m.start(), m.end()) for m in _token_pat.finditer(s)]


def span_to_token_span(tokens, cstart, cend):
    touched = [
        i for i, (_, ts, te) in enumerate(tokens) if not (te <= cstart or ts >= cend)
    ]
    if not touched:
        dists = [
            (i, min(abs(ts - cstart), abs(te - cend)))
            for i, (_, ts, te) in enumerate(tokens)
        ]
        if not dists:
            return (0, 0)
        i = min(dists, key=lambda x: x[1])[0]
        return (i, i + 1)
    return (touched[0], touched[-1] + 1)


def norm_space(s):
    return re.sub(r"\s+", " ", str(s).strip())


def normalize_for_match(s):
    s = s.lower()
    s = re.sub(r"[“”\"'`´’]", "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*-\s*", "-", s)
    return s.strip()


def _needle_to_regex(needle: str) -> re.Pattern:
    n = unicodedata.normalize("NFC", needle.strip())
    pat = re.escape(n).replace(r"\ ", r"\s+").replace(r"\-", r"[-\u2011\u2013\u2014]?")
    pat = pat.replace(r"\"", r"[\"“”]?").replace(r"\'", r"[\'’´`]?")
    return re.compile(pat, flags=re.IGNORECASE)


def tolerant_regex(needle: str) -> re.Pattern:
    n = unicodedata.normalize("NFKD", needle).replace("ß", "ss")
    n = "".join(ch for ch in n if not unicodedata.combining(ch))
    pat = re.escape(n).replace(r"\ ", r"\s+").replace(r"\-", r"[-\u2011\u2013\u2014]?")
    return re.compile(pat, flags=re.IGNORECASE)


def find_occurrence_debug(hay, needle, used):
    if not isinstance(needle, str) or not needle.strip():
        return None, "empty_needle", None, None
    pat = _needle_to_regex(needle)
    for m in pat.finditer(hay):
        span = (m.start(), m.end())
        if all(span[1] <= u[0] or span[0] >= u[1] for u in used):
            return span, None, pat.pattern, None
    diag = None
    if DIAGNOSTIC_TOLERANT:
        H = unicodedata.normalize("NFKD", hay).replace("ß", "ss")
        H = "".join(ch for ch in H if not unicodedata.combining(ch))
        if tolerant_regex(needle).search(H):
            diag = "no_match_but_tolerant_hit"
    return None, "no_match", pat.pattern, diag


# ---------- prompts ----------
SYS_PROMPT = (
    "You are a precise biomedical term aligner. Given a source and target sentence, "
    "identify (bio)medical or scientific, domain-specific terms (single or multiword: diseases, syndromes, procedures, drugs, genes, "
    "pathologies, anatomy) that appear in the source, and return their exact translations that appear "
    "in the target. Only include pairs if both surface forms appear verbatim in their respective sentences. "
    "Output strictly JSON with a 'pairs' array of objects: "
    '{"pairs":[{"src":"...","tgt":"..."}, ...]} and nothing else.'
)


def _parse_pairs(text: str):
    try:
        data = json.loads(text)
    except Exception:
        t = text.strip()
        if t.startswith("```"):
            t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
        if "{" in t and "}" in t:
            t = t[t.find("{") : t.rfind("}") + 1]
        try:
            data = json.loads(t)
        except Exception:
            return []
    pairs = data.get("pairs", []) if isinstance(data, dict) else []
    out = []
    for p in pairs:
        src_t, tgt_t = norm_space(p.get("src", "")), norm_space(p.get("tgt", ""))
        if src_t and tgt_t:
            out.append({"src": src_t, "tgt": tgt_t})
    return out


def append_jsonl(path, rec):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning(f"Failed to append to {path}: {e!r}")


# ---------- model calls ----------
def deepseek_extract_pairs_sync(
    client, src, tgt, src_lang, tgt_lang, retries=3, sleep_s=2.0
):
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
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                temperature=0.2,
                response_format={"type": "json_object"},
                max_tokens=512,
            )
            text = resp.choices[0].message.content.strip()
            pairs = _parse_pairs(text)
            if RAW_MODEL_LOG:
                append_jsonl(
                    RAW_MODEL_LOG,
                    {
                        "mode": "sync",
                        "text": text,
                        "parsed_pairs": pairs,
                        "src": src,
                        "tgt": tgt,
                    },
                )
            return pairs
        except Exception as e:
            if attempt == retries - 1:
                log.warning(f"DeepSeek sync call failed after retries: {e!r}")
                if RAW_MODEL_LOG:
                    append_jsonl(
                        RAW_MODEL_LOG,
                        {
                            "mode": "sync_error",
                            "error": repr(e),
                            "src": src,
                            "tgt": tgt,
                        },
                    )
                return []
            time.sleep(sleep_s)
    return []


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
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                temperature=0.2,
                response_format={"type": "json_object"},
                max_tokens=512,
            )
        text = resp.choices[0].message.content.strip()
        pairs = _parse_pairs(text)
        if RAW_MODEL_LOG:
            append_jsonl(
                RAW_MODEL_LOG,
                {
                    "mode": "async",
                    "text": text,
                    "parsed_pairs": pairs,
                    "src": src,
                    "tgt": tgt,
                },
            )
        return pairs

    return _call


# ---------- annotations / eval ----------
def build_annotation(idx, pid, src, tgt, pairs):
    src_tokens = tokens_with_offsets(src)
    tgt_tokens = tokens_with_offsets(tgt)
    used_src, used_tgt, term_pairs, rejected = [], [], [], []

    for p in pairs:
        s_term, t_term = p["src"], p["tgt"]
        s_span, s_err, s_pat, s_diag = find_occurrence_debug(src, s_term, used_src)
        t_span, t_err, t_pat, t_diag = find_occurrence_debug(tgt, t_term, used_tgt)
        if not s_span or not t_span:
            if DEBUG_REJECTIONS:
                rejected.append(
                    {
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
                    }
                )
            continue
        used_src.append(s_span)
        used_tgt.append(t_span)
        s_ts, s_te = span_to_token_span(src_tokens, *s_span)
        t_ts, t_te = span_to_token_span(tgt_tokens, *t_span)
        term_pairs.append(
            {
                "src": {
                    "text": s_term,
                    "char_start": s_span[0],
                    "char_end": s_span[1],
                    "token_start": s_ts,
                    "token_end": s_te,
                },
                "tgt": {
                    "text": t_term,
                    "char_start": t_span[0],
                    "char_end": t_span[1],
                    "token_start": t_ts,
                    "token_end": t_te,
                },
            }
        )
    ann = {"idx": idx, "pair_id": pid, "src": src, "tgt": tgt, "term_pairs": term_pairs}
    if DEBUG_REJECTIONS:
        ann["debug_rejections"] = rejected
    return ann


def exact_match_terms(tgt_terms, pred_sent):
    pred_norm = normalize_for_match(pred_sent)
    results, hit = [], 0
    for t in tgt_terms:
        ok = normalize_for_match(t) in pred_norm
        results.append({"term": t, "match": ok})
        if ok:
            hit += 1
    denom = len(tgt_terms) if tgt_terms else 0
    acc = (hit / denom) if denom else None
    return {"matched": hit, "total": denom, "accuracy": acc, "details": results}


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


# ---------- file-based corpus ----------
def read_ids(corpus_file, max_examples=None):
    ids = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            ids.append(s)
            if max_examples and len(ids) >= max_examples:
                break
    return ids


def read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None
    except Exception:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()


def iter_pairs_from_files(ids, data_dir, src, tgt, filename_template):
    for pid in ids:
        src_path = os.path.join(data_dir, filename_template.format(id=pid, lang=src))
        tgt_path = os.path.join(data_dir, filename_template.format(id=pid, lang=tgt))
        src_txt = read_text(src_path)
        tgt_txt = read_text(tgt_path)
        yield pid, src_txt, tgt_txt, src_path, tgt_path


# ---------- pipelines ----------
def extract_and_annotate_files_sync(args):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    ids = read_ids(args.corpus_file, args.max_examples)
    out = []
    for i, (pid, src_txt, tgt_txt, sp, tp) in enumerate(
        iter_pairs_from_files(
            ids, args.data_dir, args.src, args.tgt, args.filename_template
        )
    ):
        if (
            not isinstance(src_txt, str)
            or not isinstance(tgt_txt, str)
            or not src_txt.strip()
            or not tgt_txt.strip()
        ):
            log.warning(f"idx={i} missing/empty text for id={pid} (src={sp}, tgt={tp})")
            ann = {
                "idx": i,
                "pair_id": pid,
                "src": norm_space(src_txt),
                "tgt": norm_space(tgt_txt),
                "term_pairs": [],
            }
            if DEBUG_REJECTIONS:
                ann["debug_rejections"] = [
                    {"src_term": "", "tgt_term": "", "src_error": "empty_src_or_tgt"}
                ]
            out.append(ann)
            continue
        pairs = deepseek_extract_pairs_sync(
            client,
            src_txt,
            tgt_txt,
            args.src_lang,
            args.tgt_lang,
            retries=args.retries,
            sleep_s=args.retry_sleep,
        )
        out.append(build_annotation(i, pid, src_txt, tgt_txt, pairs))
    if args.out_annotations:
        write_jsonl(args.out_annotations, out)
    return out


async def extract_and_annotate_files_async(args):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    aclient = AsyncOpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    ids = read_ids(args.corpus_file, args.max_examples)
    items = list(
        iter_pairs_from_files(
            ids, args.data_dir, args.src, args.tgt, args.filename_template
        )
    )
    limiter = AsyncLimiter(args.rate_limit, time_period=1)
    caller = make_async_caller(
        async_client=aclient,
        limiter=limiter,
        retries=args.retries,
        retry_min=max(0.5, args.retry_sleep / 2),
        retry_max=max(2.0, args.retry_sleep * 4),
    )
    sem = asyncio.Semaphore(args.concurrency)
    results = [None] * len(items)

    async def worker(i: int, pid, src_txt, tgt_txt, sp, tp):
        async with sem:
            if (
                not isinstance(src_txt, str)
                or not isinstance(tgt_txt, str)
                or not src_txt.strip()
                or not tgt_txt.strip()
            ):
                log.warning(
                    f"idx={i} missing/empty text for id={pid} (src={sp}, tgt={tp})"
                )
                rec = {
                    "idx": i,
                    "pair_id": pid,
                    "src": norm_space(src_txt),
                    "tgt": norm_space(tgt_txt),
                    "term_pairs": [],
                }
                if DEBUG_REJECTIONS:
                    rec["debug_rejections"] = [
                        {
                            "src_term": "",
                            "tgt_term": "",
                            "src_error": "empty_src_or_tgt",
                        }
                    ]
                results[i] = rec  # type: ignore
                return
            try:
                pairs = await caller(src_txt, tgt_txt, args.src_lang, args.tgt_lang)
            except Exception as e:
                log.warning(f"idx={i} id={pid} failed: {e!r}")
                pairs = []
            results[i] = build_annotation(i, pid, src_txt, tgt_txt, pairs)  # type: ignore

    tasks = [worker(i, *items[i]) for i in range(len(items))]
    await tqdm_asyncio.gather(*tasks, desc="Extracting term pairs (concurrent)")
    out = [r for r in results if r is not None]
    if args.out_annotations:
        write_jsonl(args.out_annotations, out)
    return out


def evaluate_predictions(args, annotations=None):
    if annotations is None:
        if not args.out_annotations:
            raise RuntimeError(
                "Annotations not provided; specify --out_annotations or pass in-memory data."
            )
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
        eval_lines.append(
            {
                "idx": idx,
                "pair_id": rec.get("pair_id"),
                "prediction": pred,
                "target_terms": tgt_terms,
                "matched": em["matched"],
                "total": em["total"],
                "accuracy": em["accuracy"],
                "details": em["details"],
            }
        )
    summary = {
        "micro_accuracy": (total_hit / total_terms) if total_terms else None,
        "total_terms": total_terms,
        "total_matched": total_hit,
        "num_examples": len(annotations),
    }
    if args.out_eval:
        write_jsonl(args.out_eval, eval_lines)
        with open(args.out_eval + ".summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return {"summary": summary, "per_example": eval_lines}


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(
        description="File-based biomedical term extraction + exact-match evaluation"
    )
    # file-based inputs
    p.add_argument(
        "--corpus_file", required=True, help="Text file with one ID per line"
    )
    p.add_argument(
        "--data_dir", required=True, help="Folder containing <id>_<lang>.txt files"
    )
    p.add_argument(
        "--src", required=True, help="Source lang code used in filenames (e.g., en)"
    )
    p.add_argument(
        "--tgt", required=True, help="Target lang code used in filenames (e.g., de)"
    )
    p.add_argument(
        "--filename_template",
        default="{id}_{lang}.txt",
        help="Filename template with {id} and {lang} placeholders",
    )

    # model-language hints (for prompt only)
    p.add_argument(
        "--src_lang", default="English", help="Human-readable source language name"
    )
    p.add_argument(
        "--tgt_lang", default="German", help="Human-readable target language name"
    )
    p.add_argument("--max_examples", type=int, default=None)

    # outputs
    p.add_argument("--out_annotations", default="annotations.jsonl")
    p.add_argument(
        "--predictions",
        default=None,
        help='JSONL with {"prediction": "..."} per line, aligned to corpus order',
    )
    p.add_argument("--out_eval", default="eval_terms.jsonl")

    # retries / concurrency
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry_sleep", type=float, default=2.0)
    p.add_argument("--concurrency", type=int, default=8, help="Max in-flight requests")
    p.add_argument("--rate_limit", type=int, default=15, help="Requests per second")
    p.add_argument("--no_async", action="store_true", help="Force synchronous mode")

    # debugging
    p.add_argument(
        "--debug_rejections",
        action="store_true",
        help="Include per-example rejection reasons",
    )
    p.add_argument(
        "--raw_model_log", default=None, help="Append raw model outputs here (JSONL)"
    )
    p.add_argument(
        "--diagnostic_tolerant",
        action="store_true",
        help="Run tolerant secondary match to flag near-misses",
    )
    return p.parse_args()


def main():
    global DEBUG_REJECTIONS, RAW_MODEL_LOG, DIAGNOSTIC_TOLERANT
    args = parse_args()
    DEBUG_REJECTIONS = args.debug_rejections
    RAW_MODEL_LOG = args.raw_model_log
    DIAGNOSTIC_TOLERANT = args.diagnostic_tolerant

    if args.no_async or args.concurrency <= 1:
        annotations = extract_and_annotate_files_sync(args)
    else:
        annotations = asyncio.run(extract_and_annotate_files_async(args))

    if args.predictions:
        result = evaluate_predictions(args, annotations)
        print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    else:
        print(f"Wrote {len(annotations)} annotations to {args.out_annotations}")


if __name__ == "__main__":
    main()
