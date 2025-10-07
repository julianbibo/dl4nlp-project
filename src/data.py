from torch.utils.data import Dataset, Subset, ConcatDataset
import os
import copy
from collections import Counter
from random import Random
from typing import Tuple, Set, Sequence, Literal, List, Union, Optional
import json

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


class TranslationDataset(Dataset):
    """A natural translation dataset."""

    # TODO: make split be used everywhere
    def __init__(
        self, lang_from: str, lang_to: str, folder, valid_langs: Set[str], split: Literal["train", "eval", None] = None
    ):
        """
        Loads a dataset from the medline corpus 2022.
        Samples will be in the language specified by `lang_from`
        and labels in the language `lang_to`. One language must be 'en' (English),
        the other one of {'es' (Spanish), 'fr' (French), 'pt' (Portuguese), 'de' (German), 'it' (Italian), 'ru' (Russian)}.

        If the split is 'train' or 'eval', a .txt file with the corresponding document ID is searched for. 
        If not present, a `ValueError` is thrown. If the split is `None`, all files in the dataset are included.

        Reads from the directory {folder}/en_{other language} (e.g., wmt22/en_pt). Assumes file names within that directory
        follow the following convention: {file id}_{language}.txt (e.g., for file ID 120 we need files 120_en.txt and 120_pt.txt for english to/from portuguese).
        """

        SHUFFLE_SEED = 391

        assert lang_from in valid_langs, (
            f"Specified language '{lang_from}' is not valid! (must be one of {valid_langs})"
        )
        assert lang_to in valid_langs, (
            f"Specified language '{lang_to}' is not valid! (must be one of {valid_langs})"
        )
        assert lang_from == "en" or lang_to == "en", (
            "One of the languages must be english!"
        )
        assert lang_from != lang_to, "The from and to language may not be the same!"

        assert split in { "train", "eval", None }, (
            f"Specified split '{split}' is not valid! Must be one of {{ 'train', 'eval', None }}"
        )

        # the language that is not english
        other_lang = lang_from if lang_from != "en" else lang_to

        self.data_dir = os.path.join(folder, f"en_{other_lang}")

        # load entire dataset
        if split is None:
            # load file IDs from path
            ids = []
            for file in os.listdir(self.data_dir):
                ids.append(file.split("_")[0])

            # filter entries that don't have a translation
            ids = [idx for idx, count in Counter(ids).items() if count == 2]

            print(f"Loading dataset by extracting doc IDs in '{self.data_dir}' (n={len(ids)})...")
        # load specific split
        else:
            # check if file containing IDs corresponding to split exists 
            ids_path = os.path.join(folder, f"en_{other_lang}_{split}_ids.txt")
            if os.path.exists(ids_path):
                with open(ids_path, "r") as f:
                    ids = f.read().splitlines()
            else:
                raise ValueError(f"Could not find IDs for split '{split}'")

            print(f"Loading dataset using doc IDs found in '{ids_path}' (n={len(ids)})...")

        # shuffle deterministically
        self.ids = sorted(ids)
        Random(SHUFFLE_SEED).shuffle(self.ids)

        self.lang_from = lang_from
        self.lang_to = lang_to
        self.valid_langs = valid_langs

    # TODO: remove everywhere
    def train_test_split(self, train_perc=0.9) -> Tuple[Subset, Subset]:
        """
        Returns a deterministic train/test split.
        """

        total_samples = len(self)
        train_samples = int(total_samples * train_perc)

        return Subset(self, list(range(train_samples))), Subset(
            self, list(range(train_samples, total_samples))
        )

    def select(self, doc_ids: Sequence) -> Subset:
        """
        Returns a subset with the selected indices. May fail if `inds` contains
        invalid indices.
        """

        ids = [self.ids.index(idx) for idx in doc_ids]
        return Subset(self, ids)

    def __len__(self) -> int:
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


class Medline(TranslationDataset):
    """MEDLINE dataset used for training and evaluation, which consists of abstracts of biomedical papers."""

    def __init__(self, lang_from, lang_to, folder, split: Literal["train", "eval", None] = None):
        VALID_LANGS = { "es", "fr", "pt", "de", "it", "ru", "en" }

        super().__init__(lang_from, lang_to, folder, VALID_LANGS, split)


class GeneralTrainDataset(TranslationDataset):
    """WMT24++ dataset used for training, which consists of general-domain data."""

    def __init__(self, lang_from, lang_to, folder):
        VALID_LANGS = { "fr", "de", "it", "ru", "en" }

        # split is None, since we're only using this dataset for training
        super().__init__(lang_from, lang_to, folder, VALID_LANGS, split=None)


def load_super_train_dataset(
    target_en: bool, biomedical_lang: Optional[str], wmt22_folder, wmt24pp_folder
) -> ConcatDataset:
    """
    Loads a dataset containing a mix of general (`GeneralTrainDataset`) and biomedical datasets (`Medline`) used for training.
    Contains translations between { "fr", "de", "it", "ru" } 
    and "en". If `target_en` is `True`, then English is the target language, otherwise it's
    the source language.

    By default, all language pairs are represented by general-domain data from the WMT24++ dataset.
    If `biomedical_lang` is specified, that language to/from English is represented
    by biomedical data from the WMT22 dataset.
    """

    LANGS = { "fr", "de", "it", "ru" }

    # * load datasets *
    datasets = []

    print("Loading SuperTrainDataset...")

    for lang in LANGS:
        if target_en:
            lang_from, lang_to = lang, "en"
        else:
            lang_from, lang_to = "en", lang

        if biomedical_lang is not None and biomedical_lang == lang:
            # load biomedical data
            datasets.append(
                Medline(lang_from, lang_to, wmt22_folder, split="train")
            )

            print(f"Loaded Medline dataset for {lang_from} -> {lang_to} with {len(datasets[-1])} samples...")
        else:
            # load general data
            datasets.append(
                GeneralTrainDataset(lang_from, lang_to, wmt24pp_folder)
            )

            print(f"Loaded WMT24++ dataset for {lang_from} -> {lang_to} with {len(datasets[-1])} samples...")

    return ConcatDataset(datasets)

class MatraEvalDataset(TranslationDataset):
    """Term-to-term dictionary dataset used for evaluation, which consists of biomedical terms."""

    def __init__(self, lang_from, lang_to, folder):
        VALID_LANGS = {"es", "de", "nl", "fr"}

        # split is None, since we're only using this dataset for evaluation
        super().__init__(lang_from, lang_to, folder, VALID_LANGS, split=None)

class BiomedTermsDataset(Dataset):

    def __init__(self, annotations_jsonl_path: str,
                 unique: bool = True,
                 lower: bool = False,
                 include_meta: bool = False):
        self.include_meta = include_meta
        pairs = []
        seen = set()

        with open(annotations_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                src_sent, tgt_sent = rec.get("src", ""), rec.get("tgt", "")
                for p in rec.get("term_pairs", []):
                    s = p["src"]["text"]
                    t = p["tgt"]["text"]
                    s_ = s.lower() if lower else s
                    t_ = t.lower() if lower else t
                    key = (s_, t_)
                    if unique and key in seen:
                        continue
                    seen.add(key)
                    if include_meta:
                        meta = {
                            "idx": rec.get("idx"),
                            "pair_id": rec.get("pair_id"),
                            "src_sentence": src_sent,
                            "tgt_sentence": tgt_sent,
                            "src_span_char": (p["src"]["char_start"], p["src"]["char_end"]),
                            "tgt_span_char": (p["tgt"]["char_start"], p["tgt"]["char_end"]),
                            "src_span_tok": (p["src"]["token_start"], p["src"]["token_end"]),
                            "tgt_span_tok": (p["tgt"]["token_start"], p["tgt"]["token_end"]),
                        }
                        pairs.append((s_, t_, meta))
                    else:
                        pairs.append((s_, t_))

        self._data = pairs

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]



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
    toks = tokenizer.pad(
        {"input_ids": source_toks}, padding=True, return_tensors="pt"
    ).to(device)

    # for labels, set prompt tokens to IGNORE_TOKEN as we don't want to compute loss over those
    toks["labels"] = copy.deepcopy(toks["input_ids"])

    for i in range(len(toks["labels"])):
        toks["labels"][i][: len(source_toks_raw[i]) + len(bos)] = IGNORE_TOKEN

    return toks
