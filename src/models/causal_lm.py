from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List

class NMTModel:
    def __init__(self, model_dir: str, device, role: Optional[str] = None):
        """
        Loads neural machine translation model.

        # Args
        * `model_dir`: Directory that includes the model, config, and tokenizer.
        * `role`: Description of the system's role, if `None` then the default role is chosen.
        """

        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype="auto",
            device_map=device,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        if role is None:
            self.role = "You are a helpful translation assistant."
        else:
            self.role = role

    def prompt(self, prompt: str) -> str:
        return self.prompt_batch([prompt])

    def prompt_batch(self, prompts: List[str]) -> List[str]:
        """
        Runs inference.
        """

        text_batch = [
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": self.role}, {"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            ) for prompt in prompts
        ]
        model_inputs = self.tokenizer(text_batch, return_tensors="pt", padding=True).to(self.model.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            attention_mask=model_inputs.attention_mask,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    