from torch.utils.data import Dataset
import os
import copy
from typing import Tuple

IGNORE_TOKEN = -100  # TODO: set in loss function


def full_lang_name(abbr: str):
    """
    Returns the full name of the language from its abbrieviation.
    """

    NAMES = {
        "es": "Spanish",
        "fr": "French",
        "pt": "Portuguese",
        "de": "German",
        "it": "Italian",
        "ru": "Russian",
        "en": "English",
    }

    return NAMES[abbr]


class Medline(Dataset):
    def __init__(self, lang_from: str, lang_to: str, folder: str):
        """
        Loads biomedical dataset from the medline corpus 2022.
        Samples will be in the language specified by `lang_from`
        and labels in the language `lang_to`. One language must be 'en' (English),
        the other one of {'es' (Spanish), 'fr' (French), 'pt' (Portuguese), 'de' (German), 'it' (Italian), 'ru' (Russian)}.

        Reads from the directory {folder}/en_{other language} (e.g., wmt22/en_pt). Assumes file names within that directory
        follow the following convention: {file id}_{language}.txt (e.g., for file ID 120 we need files 120_en.txt and 120_pt.txt for english to/from portuguese).
        """

        VALID_LANGS = {"es", "fr", "pt", "de", "it", "ru", "en"}

        assert lang_from in VALID_LANGS, (
            f"Specified language '{lang_from}' is not valid! (must be one of {VALID_LANGS})"
        )
        assert lang_to in VALID_LANGS, (
            f"Specified language '{lang_to}' is not valid! (must be one of {VALID_LANGS})"
        )
        assert lang_from == "en" or lang_to == "en", (
            "One of the languages must be english!"
        )
        assert lang_from != lang_to, "The from and to language may not be the same!"

        # the language that is not english
        other_lang = lang_from if lang_from != "en" else lang_to

        self.data_dir = os.path.join(folder, f"en_{other_lang}")

        # load file IDs
        self.ids = []
        for file in os.listdir(self.data_dir):
            self.ids.append(file.split("_")[0])

        self.lang_from = lang_from
        self.lang_to = lang_to

    def __len__(self):
        return len(self.ids)

    def _read_file(self, idx, lang) -> str:
        """
        Tries to read the contents of a language file.
        # Errors
        * If the file corresponding to `idx` and `lang` does not exist.
        """

        path = os.path.join(self.data_dir, f"{idx}_{lang}.txt")

        with open(path, "r") as file:
            return file.read().rstrip()

    def __getitem__(self, index) -> Tuple[str, str]:
        """
        Fetches a source and target.
        """

        idx = self.ids[index]

        source = self._read_file(idx, self.lang_from)
        target = self._read_file(idx, self.lang_to)

        return source, target


def get_translation_prompt_skeleton(lang_from: str, lang_to: str) -> str:
    """
    Returns a translation prompt skeleton with the source yet to be plugged in.
    The format is adapted from "How Good Are GPT Models at Machine Translation? A Comprehensive Evaluation".
    # Form
    `Translate this from <lang 0> to <lang 1>: <lang 0>: <source> <lang 1>:`
    """
    return (
        f"Translate this from {lang_from} to {lang_to}: {lang_from}: {{}} {lang_to}: "
    )


def collate_translations(batch, tokenizer, prompt_form: str, device):
    """
    Returns a dictionary containing input_ids, labels, and attention_mask entries
    by taking a batch of string source target pairs, formatting them into a translation prompt,
    and tokenizing.
    """

    source = [prompt_form.format(s[0]) for s in batch]
    target = [s[1] for s in batch]

    source_toks_raw = tokenizer(source, add_special_tokens=False)["input_ids"]
    target_toks_raw = tokenizer(target, add_special_tokens=False)["input_ids"]

    # build tokens
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    eos = [tokenizer.eos_token_id]

    source_toks = [bos + s + t + eos for s, t in zip(source_toks_raw, target_toks_raw)]
    toks = tokenizer.pad({"input_ids": source_toks}, padding=True, return_tensors="pt")
    toks = toks.to(device)

    # for labels, set prompt tokens to IGNORE_TOKEN as we don't want to compute loss over those
    toks["labels"] = copy.deepcopy(toks["input_ids"])

    for i in range(len(toks["labels"])):
        toks["labels"][i][: len(source_toks_raw[i])] = IGNORE_TOKEN
    return toks
