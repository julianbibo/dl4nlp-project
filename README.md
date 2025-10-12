# Multilingual Domain-specific Finetuning

This is the repository for our paper "Multilingual Domain-specific Finetuning" (Stan Fris, Fabian Westerbeek, Morris de Haan, Julian Bibo, Quinten van Engelen).

## Abstract

This repository contains the code for our study on domain-specific fine-tuning of Multilingual Neural Machine Translation (MNMT) models. We fine-tune Qwen2-1.5B-Instruct on biomedical subsets of the WMT’22 Biomedical Translation corpus and evaluate performance on both WMT’22 and Mantra datasets. We introduce the Term Matching Score (TMS) to measure domain terminology accuracy. Results show that fine-tuning improves general translation quality across most language pairs, with domain fluency driving cross-language transfer more than linguistic similarity.

## Setup

1. Create a virtual environment and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```
2. Install the required packages. The requirements are set up for Python 3.9.18.

```bash
pip install -r requirements.txt
```

3. Setup the environment variables in `environment/.env`. You can copy the example file:

```bash
cp environment/.env.example environment/.env
```

4. Download the [Mantra](https://huggingface.co/datasets/bigbio/mantra_gsc), [WMT24++](https://arxiv.org/abs/2502.12404v1), and [WMT22](https://github.com/biomedical-translation-corpora/corpora?tab=readme-ov-file) datasets and place them in the `data/` directory, and preprocess if necessary.


## Usage

You can run the finetuning and evaluation jobs located in `jobs/`

```bash
sbatch jobs/finetune.job # for finetuning
sbatch jobs/eval.job <model> <source_lang> <target_lang> <annotation_lang> <data_dir> # for evaluation
```

