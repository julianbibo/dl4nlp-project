from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from typing import Tuple, Union


def load(model_name: str, device, dtype) -> Tuple[Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM], AutoTokenizer]:
    """
    Loads a model and tokenizer.
    # Errors
    * Raises a `NotImplementedError` if the given model name is not supported.
    """
    
    model_name = model_name.lower()

    if model_name in {"qwen2-7b"}:
        model_name = "Qwen/Qwen2-7B"
        model_cls = AutoModelForCausalLM

    elif model_name in {"qwen2-0.5b", "qwen2-0.5b-instruct"}:
        model_name = "Qwen/Qwen2-0.5B-Instruct"
        model_cls = AutoModelForCausalLM

    elif model_name in {"nllb-200-distilled-600m"}:
        model_name = "facebook/nllb-200-distilled-600M"
        model_cls = AutoModelForSeq2SeqLM

    elif model_name in {
        "llama-3.2-3b",
        "llama-3-3b",
        "llama-3.2-3b-instruct",
        "llama-3-3b-instruct",
    }:
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        model_cls = AutoModelForCausalLM

    elif model_name in {"qwen2-1.5b"}:
        model_name = "Qwen/Qwen2-1.5B"
        model_cls = AutoModelForCausalLM

    else:
        raise NotImplementedError(f"Model {model_name} not supported.")

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = model_cls.from_pretrained(
        model_name,
        device_map=device,
        dtype=dtype,
    )

    return model, tokenizer