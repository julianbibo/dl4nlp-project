import torch
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, random_split
from functools import partial
from data import Medline, collate_translations, get_translation_prompt_skeleton, full_lang_name
from models.causal_lm import NMTModel


class EncoderTrainer(Trainer):
    """
    Custom trainer for optimizing encoder-only NMT transformer models.

    During training and evaluation, it provides source sentences (possibly wrapped in a prompt to indicate the type of task to the model)
    and optimizes the model for predicting correct target sentences. In other words, loss or metrics are not calculated over the
    source prompt, only over the target sentence.
    """

    def __init__(self, prompt_skeleton: str, tokenizer, **kwargs):
        super().__init__(**kwargs)

        self.collate_fn = partial(collate_translations, tokenizer=tokenizer, prompt_form=prompt_skeleton)

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=self.collate_fn
        )
    
    def get_eval_dataloader(self, eval_dataset = None):
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        return DataLoader(
            dset[eval_dataset] if isinstance(eval_dataset, str) else eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=dset.build_collator_fn
        )
    
    def get_test_dataloader(self, test_dataset):
        return DataLoader(
            test_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.collate_fn
        )    

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> float:
    # TODO:
    pass

def parse_args():
    # TODO:
    pass


if __name__ == "__main__":
    args = parse_args()

    # * load model *
    model = NMTModel(args.model_dir, args.device)

    # * load data *
    dset = Medline("de", "en", "../data/wmt22")
    train_dset, test_dset = random_split(dset, [0.8, 0.2])

    # * load finetuneable model *
    # TODO: find good parameters
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    model_peft = get_peft_model(model.model, peft_config)

    model_peft.print_trainable_parameters()

    # * setup trainer *
    # TODO: find good parameters
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        learning_rate=1e-3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
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
        prompt_skeleton=get_translation_prompt_skeleton(full_lang_name(dset.lang_from), full_lang_name(dset.lang_to)),
        tokenizer=model.tokenizer,

        model=model,
        args=training_args,
        train_dataset=train_dset,
        eval_dataset=test_dset,
        processing_class=model.tokenizer,
        compute_metrics=compute_metrics,
    )

    # * train *
    trainer.train()