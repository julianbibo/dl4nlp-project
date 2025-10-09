import argparse
import os
from uuid import uuid4
import time
import json
import re

import torch
import evaluate
from transformers import __version__ as transformers_version
from tqdm import tqdm
from dotenv import load_dotenv

import models
from data import (
    Medline,
    get_translation_prompt_skeleton,
    full_lang_name,
)

CSV_HEADER = "job_id,model,data_dir,source_lang,target_lang,lp,chrf,comet,sacrebleu,term_em_micro"
load_dotenv("./environment/.env")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
SLURM_JOB_ID = os.getenv("SLURM_JOB_ID") or "local-" + uuid4().hex[:8]
if not os.path.exists("results.csv"):
    with open("results.csv", "w") as f:
        f.write(f"{CSV_HEADER}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        "Evaluate translation model (Comet, CometKiwi, chrF++, exact term EM)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="HF model ID or path.",
    )
    parser.add_argument("--data_dir", type=str, help="Path to dataset.")
    parser.add_argument(
        "--split",
        type=str,
        help="Dataset partition: {'test', 'train'}.",
        choices=["test", "train"],
    )
    parser.add_argument("--source_lang", type=str)
    parser.add_argument("--target_lang", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument(
        "--length_penalties",
        default="0.8",
        help="Comma-separated values, e.g. 0.6,0.8,1.0",
    )
    parser.add_argument("--comet_model", default="Unbabel/wmt22-comet-da")
    parser.add_argument("--cometkiwi_model", default="Unbabel/wmt22-cometkiwi-da")
    parser.add_argument(
        "--annotations_jsonl",
        type=str,
        default=None,
        help="Path to annotations JSONL with 'idx' and term_pairs.",
    )
    return parser.parse_args()


# ---- helpers ----
def _read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(json.loads(s))
    return out


def _normalize_for_match(s: str) -> str:
    s = s.lower()
    s = re.sub(r'[“”"\'`´’]', "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*-\s*", "-", s)
    return s.strip()


def _exact_match_terms(tgt_terms, pred_sent):
    pred_norm = _normalize_for_match(pred_sent or "")
    hit = 0
    for t in tgt_terms:
        if _normalize_for_match(t) in pred_norm:
            hit += 1
    return hit, len(tgt_terms)


def main():
    MAX_TEST_SAMPLES = 128

    print(f"Transformers version: {transformers_version}")
    args = parse_args()
    print(f"Args: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}...")

    medline = Medline(args.source_lang, args.target_lang, args.data_dir, split="eval")

    if args.split not in ["train", "test"]:
        raise ValueError(
            f"Datasplit {args.split} not recognized! Must be on of {{'test', 'train'}}"
        )

    # derive split
    try:
        train_inds_path = os.path.join(args.model, "..", "train_doc_ids.txt")
        f = open(train_inds_path, "r")
        train_inds = f.read().splitlines()

        print("Using observed train doc IDs for split...")

        if args.split == "test":
            indices = [idx for idx in medline.ids if idx not in train_inds]
            dataset = medline.select(indices)
        elif args.split == "train":
            indices = [idx for idx in medline.ids if idx in train_inds]
            dataset = medline.select(indices)
    except FileNotFoundError:
        print("Doc IDs not found; using default split...")

        if args.split == "test":
            _, dataset = medline.train_test_split()
        elif args.split == "train":
            dataset, _ = medline.train_test_split()

    # annotations -> idx -> list of target terms
    abstract_id_to_terms = None
    if args.annotations_jsonl:
        anns = _read_jsonl(args.annotations_jsonl)
        abstract_id_to_terms = {
            rec["pair_id"]: [p["tgt"]["text"] for p in rec.get("term_pairs", [])]
            for rec in anns
        }

    # align IDs with dataset order
    # item_ids = list(getattr(dataset, "doc_id"))  # @morris im not sure NOTE: does not work, pls delete

    comet = evaluate.load("comet", revision="main")
    chrf = evaluate.load("chrf", revision="main")
    sacrebleu = evaluate.load("sacrebleu")

    model, tokenizer = models.load(
        args.model, device, args.dtype, tokenizer_padding_side="left"
    )
    model.eval()

    model = torch.compile(model.to(device))  # may speed up
    torch.set_float32_matmul_precision("high")

    prompt_skeleton = get_translation_prompt_skeleton()

    # prompts + refs
    sources, references, doc_ids = [], [], []
    for batch in dataset:  # type: ignore
        sources.append(
            prompt_skeleton.format(
                lang_from=full_lang_name(batch["lang_from"]),
                lang_to=full_lang_name(batch["lang_to"]),
                source=batch["source"],
            )
        )
        references.append(batch["target"])
        doc_ids.append(batch["doc_id"])

    length_penalties = [float(x) for x in args.length_penalties.split(",")]

    for length_penalty in length_penalties:
        print(">>> Evaluating...")
        preds = []

        n_samples = min(MAX_TEST_SAMPLES, len(sources))
        for i in tqdm(range(0, n_samples, args.batch_size)):
            prompts = sources[i : i + args.batch_size]
            max_gen_len = int(
                max([len(s) for s in references[i : i + args.batch_size]]) * 1.3
            )
            print(
                f"Batch {i}: Max generation length: {max_gen_len}, using greedy decoding..."
            )

            text_batch = [
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": "You are a helpful translation assistant.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in prompts
            ]

            model_inputs = tokenizer(  # type: ignore
                text_batch, return_tensors="pt", padding=True, truncation=True
            ).to(device)

            start_time = time.perf_counter()
            with torch.no_grad():
                generated_ids = model.generate(  # type: ignore
                    model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    max_new_tokens=max_gen_len,
                    cache_implementation="static",
                )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            decoded = tokenizer.batch_decode(  # type: ignore
                generated_ids, skip_special_tokens=True
            )
            end_time = time.perf_counter()

            preds.extend(decoded)

            if i < 10:
                for src, pred in zip(prompts, decoded):
                    print(f"SRC: {src}\nPRED: {pred}\n")

            print(f"!Iteration took {end_time - start_time}s!")

        assert len(preds) == len(references), f"Preds length: {len(preds)}, References length: {len(references)}"
        

        # exact term EM (micro)
        if abstract_id_to_terms is None:
            term_em_micro = -1
        else:
            total_hit, total_terms = 0, 0
            for pred, doc_id in zip(preds, doc_ids):
                if doc_id not in abstract_id_to_terms:
                    print(
                        f"WARNING: doc_id {doc_id} not found in annotations, skipping"
                    )
                    continue

                tgt_terms = abstract_id_to_terms[doc_id]
                h, t = _exact_match_terms(tgt_terms, pred)
                total_hit += h
                total_terms += t
            term_em_micro = (total_hit / total_terms) if total_terms else 0.0
            print(
                f"Exact-match terms (micro): {term_em_micro:.4f} "
                f"[matched={total_hit}, total={total_terms}]"
            )

        # COMET
        comet_score = comet.compute(
            predictions=preds,
            references=references,
            sources=sources,
        )["mean_score"]

        # chrF++
        chrf_score = chrf.compute(
            predictions=preds, references=[[r] for r in references]
        )["score"]  # type: ignore

        # sacreBLEU
        bleu_score = sacrebleu.compute(
            predictions=preds, references=[[r] for r in references]
        )["score"]

        print(CSV_HEADER)
        results_str = (
            f"{SLURM_JOB_ID},{args.model},{args.data_dir},{args.source_lang},"
            f"{args.target_lang},{length_penalty},{chrf_score:.3f},{comet_score:.3f},"
            f"{bleu_score:.3f},{term_em_micro:.3f}"
        )
        print(results_str)

        with open("results.csv", "a") as f:
            f.write(f"{results_str}\n")


if __name__ == "__main__":
    main()
