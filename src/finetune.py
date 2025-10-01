import os
from functools import partial
from argparse import ArgumentParser
from uuid import uuid4

import torch

# from models.causal_lm import NMTModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, random_split
from data import (
    Medline,
    collate_translations,
    get_translation_prompt_skeleton,
    full_lang_name,
)
import wandb
from dotenv import load_dotenv

load_dotenv("./environment/.env")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
SLURM_JOB_ID = os.getenv("SLURM_JOB_ID") or "local-" + uuid4().hex[:8]


print(WANDB_API_KEY, SLURM_JOB_ID)
wandb.login(key=WANDB_API_KEY)


class EncoderTrainer(Trainer):
    """
    Custom trainer for optimizing encoder-only NMT transformer models.

    During training and evaluation, it provides source sentences (possibly wrapped in a prompt to indicate the type of task to the model)
    and optimizes the model for predicting correct target sentences. In other words, loss or metrics are not calculated over the
    source prompt, only over the target sentence.
    """

    def __init__(
        self, prompt_skeleton: str, tokenizer, train_dataset, eval_dataset, **kwargs
    ):
        super().__init__(
            train_dataset=train_dataset, eval_dataset=eval_dataset, **kwargs
        )

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.collate_fn = partial(
            collate_translations,
            tokenizer=tokenizer,
            prompt_form=prompt_skeleton,
            device=self.model.device,  # type: ignore
        )

    def get_train_dataloader(self):
        return DataLoader(  # TODO: use accelerator.prepare?
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=self.collate_fn,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        return DataLoader(
            self.eval_dataset,  # [eval_dataset] if isinstance(eval_dataset, str) else eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.collate_fn,
        )

    def get_test_dataloader(self, test_dataset):
        return DataLoader(
            test_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.collate_fn,
        )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-sl",
        "--source_language",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-tl",
        "--target_language",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Output directory where the finetuned model and checkpoints will be saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the training on.",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Path to the training dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Data type for model weights (e.g., float16, bfloat16, float32).",
    )

    args = parser.parse_args()
    # log to wandb
    args.output_dir = os.path.join(args.checkpoint_dir, f"finetune-{SLURM_JOB_ID}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    print(f"Args: {args}")

    wandb.init(
        project="dl4nlp-project",
        name=f"finetune-{SLURM_JOB_ID}",
        config=dict(
            job_id=SLURM_JOB_ID,
        ),
    )
    wandb.config.update(args)

    return args


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)

    # load model
    if args.model.lower() in {"qwen2-7b"}:
        args.model = "Qwen/Qwen2-7B"
        model_cls = AutoModelForCausalLM

    elif args.model.lower() in {"qwen2-0.5b", "qwen2-0.5b-instruct"}:
        args.model = "Qwen/Qwen2-0.5B-Instruct"
        model_cls = AutoModelForCausalLM

    elif args.model.lower() in {"nllb-200-distilled-600m"}:
        args.model = "facebook/nllb-200-distilled-600M"
        model_cls = AutoModelForSeq2SeqLM

    elif args.model.lower() in {
        "llama-3.2-3b",
        "llama-3-3b",
        "llama-3.2-3b-instruct",
        "llama-3-3b-instruct",
    }:
        args.model = "meta-llama/Llama-3.2-3B-Instruct"
        model_cls = AutoModelForCausalLM

    elif args.model.lower() in {"qwen2-1.5b"}:
        args.model = "Qwen/Qwen2-1.5B"
        model_cls = AutoModelForCausalLM

    else:
        raise NotImplementedError(f"Model {args.model} not supported.")

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = model_cls.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=args.dtype,
    )

    # load data
    dataset = Medline(args.source_language, args.target_language, args.train_dir)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
    assert train_dataset and test_dataset, "Datasets may not be empty!"
    print("train/test dataset lengths:", len(train_dataset), len(test_dataset))

    # load finetuneable model
    # TODO: find good LR
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model_peft = get_peft_model(model, peft_config)

    model_peft.print_trainable_parameters()

    # setup trainer
    # TODO: find good parameters
    # TODO: maximize batch sizes
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=1e-3,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # TODO: find good parameters
    # TODO: set ignore_index in the loss function
    trainer = EncoderTrainer(
        # prompt used for aiding the model to make translations
        prompt_skeleton=get_translation_prompt_skeleton(
            full_lang_name(dataset.lang_from), full_lang_name(dataset.lang_to)
        ),
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        compute_metrics=None,
    )

    # TODO wandb logging callback for Trainer object example code
    from transformers import TrainerCallback

    # def compute_bleu(preds, labels, tokenizer):
    #     pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
    #     label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    #     return corpus_bleu(pred_str, [label_str]).score

    class WandbLoggingCallback(TrainerCallback):
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            # if "eval_preds" in kwargs:
            #     preds, labels = kwargs["eval_preds"]
            #     bleu = compute_bleu(preds, labels, self.tokenizer)
            #     wandb.log({"bleu": bleu}, step=state.global_step)
            pass

        def on_epoch_end(self, args, state, control, **kwargs):
            allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
            reserved = torch.cuda.memory_reserved() / (1024**2)  # MB
            wandb.log(
                {"cuda/allocated_mb": allocated, "cuda/reserved_mb": reserved},
                step=state.global_step,
            )

    trainer.add_callback(WandbLoggingCallback(tokenizer))

    # train
    trainer.train()
