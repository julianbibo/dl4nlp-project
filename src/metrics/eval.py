import argparse
import os
from uuid import uuid4

import torch
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import __version__ as transformers_version
from tqdm import tqdm
from dotenv import load_dotenv


load_dotenv("./environment/.env")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
SLURM_JOB_ID = os.getenv("SLURM_JOB_ID") or "local-" + uuid4().hex[:8]
os.makedirs("results", exist_ok=True)
if not os.path.exists("results.csv"):
    with open("results.csv", "w") as f:
        f.write("job_id,model,dataset,lp,chrf,comet,cometkiwi\n")


def parse_args():
    parser = argparse.ArgumentParser("Evaluate translation model (Comet, CometKiwi, chrF++)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--source_lang", required=True)
    parser.add_argument("--target_lang", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument(
        "--length_penalties",
        default="0.8",
        help="comma-separated values, e.g. 0.6,0.8,1.0",
    )
    parser.add_argument("--comet_model", default="Unbabel/wmt22-comet-da")
    parser.add_argument("--cometkiwi_model", default="Unbabel/wmt22-cometkiwi-da")
    return parser.parse_args()


def main():
    print(f"Transformers version: {transformers_version}")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_dataset(args.dataset, args.dataset_config, split=args.split)

    # metrics
    comet = evaluate.load("comet", revision="main")
    kiwi = evaluate.load("cometkiwi", revision="main")
    chrf = evaluate.load("chrf", revision="main")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()

    sources, references = [], []
    for ex in ds:
        if "translation" in ex:
            tr = ex["translation"]
            sources.append(tr[args.source_lang])
            references.append(tr[args.target_lang])
        else:
            sources.append(ex[args.source_lang])
            references.append(ex[args.target_lang])

    lps = [float(x) for x in args.length_penalties.split(",")]

    for lp in lps:
        preds = []
        for i in tqdm(range(0, len(sources), args.batch_size), desc=f"LP={lp}"):
            batch = sources[i : i + args.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(
                device
            )
            out = model.generate(
                **enc,
                num_beams=args.beam_size,
                max_length=tok.model_max_length,
                length_penalty=lp,
                early_stopping=False,
            )
            preds.extend(tok.batch_decode(out, skip_special_tokens=True))

        # chrF++
        chrf_score = chrf.compute(
            predictions=preds, references=[[r] for r in references]
        )["score"]

        # COMET
        comet_score = comet.compute(
            predictions=preds,
            references=references,
            sources=sources,
            model=args.comet_model,
        )["mean_score"]

        # COMETKiwi
        kiwi_score = kiwi.compute(
            predictions=preds, sources=sources, model=args.cometkiwi_model
        )["mean_score"]

        print(
            f"length_penalty={lp:.2f} → chrF++: {chrf_score:.3f}, COMET: {comet_score:.3f}, CometKiwi: {kiwi_score:.3f}"
        )

        # log to results.csv
        with open("results.csv", "a") as f:
            f.write(
                f"{SLURM_JOB_ID},{args.model},{args.dataset},{lp},{chrf_score:.3f},{comet_score:.3f},{kiwi_score:.3f}\n"
            )


if __name__ == "__main__":
    main()
