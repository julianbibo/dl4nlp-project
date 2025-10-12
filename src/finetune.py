import os
from functools import partial
from argparse import ArgumentParser
from uuid import uuid4

import torch

from transformers import TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from data import (
    SuperTrainDataset,
    collate_translations,
    get_translation_prompt_skeleton,
)
import models
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
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
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
        "--target_en",
        action="store_true",
        help="Whether the target language is English. If `False`, English serves as the source language.",
    )
    parser.add_argument(
        "--biomedical_lang",
        type=str,
        default=None,
        help="If set, this language is represented by biomedical data instead of general data",
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
        "--wmt24pp_data_dir",
        type=str,
        required=True,
        help="Path to the wmt24++ dataset.",
    )
    parser.add_argument(
        "--wmt22_data_dir",
        type=str,
        default=None,
        help="Path to the wmt22 dataset. Only necessary if biomedical_lang is set to a non-None value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization.",
    )

    # training params
    parser.add_argument(
        "--learn_rate", type=float, default=1e-3, help="Learning rate during training."
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to train for."
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
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=8,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout.",
    )

    args = parser.parse_args()

    if args.biomedical_lang is None:
        run_name = "baseline"
    else:
        run_name = f"domain={args.biomedical_lang}"

    if args.target_en:
        run_name += "_en=target"
    else:
        run_name += "_en=source"

    args.output_dir = os.path.join(
        args.checkpoint_dir, run_name, f"finetune_{SLURM_JOB_ID}"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Output directory: {args.output_dir}")
    print(f"Args: {args}")

    # log to wandb
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

    # set seed
    torch.manual_seed(args.seed)

    # load model and tokenizer
    model, tokenizer = models.load(args.model, args.device, args.dtype)

    # load data
    dataset = SuperTrainDataset(
        args.target_en, args.biomedical_lang, args.wmt24pp_data_dir, args.wmt22_data_dir
    )
    assert dataset, "Train dataset may not be empty!"

    print("train dataset length:", len(dataset))

    # load finetuneable model
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model_peft = get_peft_model(model, peft_config)  # type: ignore

    model_peft.print_trainable_parameters()

    # setup trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learn_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        # we manually save at the end
        save_strategy="no",
        eval_strategy="no",
        load_best_model_at_end=True,
    )

    trainer = EncoderTrainer(
        # prompt used for aiding the model to make translations
        prompt_skeleton=get_translation_prompt_skeleton(),
        tokenizer=tokenizer,
        model=model_peft,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        compute_metrics=None,
    )

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
    trainer.save_model()
