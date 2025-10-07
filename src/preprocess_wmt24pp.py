from pathlib import Path
import json
import argparse


# downloaded wmt24pp from:
# https://huggingface.co/datasets/google/wmt24pp/tree/main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess WMT24PP data")

    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/wmt24pp"),
        help="Path to WMT24PP data directory",
    )
    parser.add_argument(
        "--word_cap", type=int, default=400, help="Maximum number of words in abstract"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # for each .jsonl file in data_dir
    for jsonl_path in args.data_dir.glob("*.jsonl"):
        print(f"Processing {jsonl_path}...")
        with open(jsonl_path) as f:
            json_dicts = [json.loads(line) for line in f]

        keep_indices = []
        for i, json_dict in enumerate(json_dicts):
            if json_dict["is_bad_source"]:
                print(f"Bad source: skipping sample {i}")
                continue

            source_length = len(json_dict["source"].split())
            target_length = len(json_dict["target"].split())
            if source_length > args.word_cap or target_length > args.word_cap:
                print(f"Warning, skipping long sample {i}")
                continue

            keep_indices.append(i)

        print(f"Keeping {len(keep_indices)}/{len(json_dicts)} samples after filtering.")

        lang_pair_str = jsonl_path.name[:5].replace("-", "_")  # = en_XY
        source_lang, target_lang = lang_pair_str.split("_")
        save_dir = args.data_dir / lang_pair_str
        save_dir.mkdir(parents=True, exist_ok=True)

        for i in keep_indices:
            json_dict = json_dicts[i]

            with open(save_dir / f"{i}_{source_lang}.txt", "w") as f:
                f.write(json_dict["source"])
            with open(save_dir / f"{i}_{target_lang}.txt", "w") as f:
                f.write(json_dict["target"])


if __name__ == "__main__":
    main()
