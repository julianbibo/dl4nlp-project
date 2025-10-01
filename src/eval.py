import argparse
import os
from uuid import uuid4
import time

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


CSV_HEADER = "job_id,model,dataset,lp,chrf,comet,cometkiwi"
load_dotenv("./environment/.env")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
SLURM_JOB_ID = os.getenv("SLURM_JOB_ID") or "local-" + uuid4().hex[:8]
if not os.path.exists("results.csv"):
    with open("results.csv", "w") as f:
        f.write(f"{CSV_HEADER}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        "Evaluate translation model (Comet, CometKiwi, chrF++)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Identifier of model in Hugging Face database or path to pretrained weights and config.",
    )
    parser.add_argument("--data_dir", type=str, help="Path to dataset.")
    parser.add_argument(
        "--split",
        type=str,
        help="Dataset partition, must be on of {{'test', 'train'}}.",
    )
    parser.add_argument("--source_lang", type=str)
    parser.add_argument("--target_lang", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--dtype", type=str, default="float16")
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

    medline = Medline(
        args.source_lang, args.target_lang, args.data_dir
    )
    train_dataset, test_dataset = medline.train_test_split()
    print("dataset lengths:", len(train_dataset), len(test_dataset))

    if args.split == "test":
        dataset = test_dataset
    elif args.split == "train":
        dataset = train_dataset
    else:
        raise NotImplementedError(
            f"Datasplit {args.split} not recognized! Must be on of {{'test', 'train'}}"
        )

    # kiwi = evaluate.load("cometkiwi", revision="main")
    chrf = evaluate.load("chrf", revision="main")

    model, tokenizer = models.load(args.model, device, args.dtype, tokenizer_padding_side="left")
    # may induce a nice inference speed-up
    model = torch.compile(model)

    prompt_skeleton = get_translation_prompt_skeleton(
        full_lang_name(medline.lang_from), full_lang_name(medline.lang_to)
    )

    # store prompts and target sentences
    sources, references = [], []
    for source, target in dataset:  # type: ignore
        sources.append(prompt_skeleton.format(source))
        references.append(target)

    length_penalties = [float(x) for x in args.length_penalties.split(",")]
    for length_penalty in length_penalties:
        print(f">>> Evaluating with length penalty {length_penalty}")
        preds = []

        for i in range(0, len(sources), args.batch_size):
            prompts = sources[i : i + args.batch_size]
            max_gen_len = int(max([len(s) for s in references[i : i + args.batch_size]]) * 1.3)

            # TODO: do we need this? and also use during finetuning?
            text_batch = [
                tokenizer.apply_chat_template(
                    [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True
                ) for prompt in prompts
            ]

            model_inputs = tokenizer(  # type: ignore
                text_batch, return_tensors="pt", padding=True, truncation=True
            ).to(device)

            start_time = time.perf_counter()

            # generate outputs
            generated_ids = model.generate(  # type: ignore
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,

                max_new_tokens=max_gen_len, # TODO: is this proper?
                
                # paired with torch.compile(model), this could lead to a nice speed-up
                # cache_implementation="static",
            )

            # cut off prompt
            generated_ids = [
                output_ids[len(input_ids):]
                    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            decoded = tokenizer.batch_decode(  # type: ignore
                generated_ids, skip_special_tokens=True
            )   

            end_time = time.perf_counter()


            preds.extend(decoded)  # type: ignore
            # print decoded msg
            # for src, pred in zip(prompts, decoded):
            #     print(f"SRC: {src}\nPRED: {pred}\n")

            print(f"!Iteration took {end_time - start_time}s!")
            break

        # chrF++
        chrf_score = chrf.compute(
            predictions=preds, references=[[r] for r in references[:len(preds)]]
        )["score"]  # type: ignore

        # TODO: compute BLEU as well
        # TODO: we're not using length penalty right now
        print(CSV_HEADER)
        results_str = f"{SLURM_JOB_ID},{args.model},{length_penalty},{chrf_score:.3f},{-1},{-1}"
        print(results_str)

        # log to results.csv
        with open("results.csv", "a") as f:
            f.write(f"{results_str}\n")


if __name__ == "__main__":
    main()
