from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from peft import AutoPeftModelForCausalLM
from typing import Tuple, Union


def load(
    # TODO: set default padding side to None
    model_name: str, device, dtype, tokenizer_padding_side="right"
) -> Tuple[Union[AutoModelForCausalLM, AutoPeftModelForCausalLM, AutoModelForSeq2SeqLM], AutoTokenizer]:
    """
    Loads a model and tokenizer.
    # Errors
    * Raises a `NotImplementedError` if the given model name is not supported.
    """

    if "nllb" in model_name.lower():
        model_cls = AutoModelForSeq2SeqLM
    elif "finetune" in model_name:
        model_cls = AutoPeftModelForCausalLM
    else:
        model_cls = AutoModelForCausalLM

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=tokenizer_padding_side)

    model = model_cls.from_pretrained(
        model_name,
        device_map=device,
        dtype=dtype,
    )

    return model, tokenizer  # type: ignore
