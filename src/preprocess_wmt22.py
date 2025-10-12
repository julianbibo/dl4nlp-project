import os
from argparse import ArgumentParser
from data import Medline
from random import Random


def parse_args():
    parser = ArgumentParser()

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
        "--data_dir",
        type=str,
        required=True,
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--word_cap",
        type=int,
        required=True,
        help="Abstract maximum length.",
    )
    parser.add_argument(
        "--train_size",
        default=960,  # lowest number of sample in wmt24
        help="Number of samples in train set.",
    )
    parser.add_argument(
        "--eval_size",
        default=128,
        help="Number of samples in evaluation set.",
    )
    parser.add_argument(
        "--seed",
        default=381,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # load entire dataset
    medline = Medline(
        args.source_language, args.target_language, args.data_dir, split=None
    )

    doc_ids = []
    for i in range(len(medline)):
        doc_id = medline.ids[i]
        source, target = medline[i]

        # filter entries that are too long
        if (
            len(source.split(" ")) <= args.word_cap
            and len(target.split(" ")) <= args.word_cap
        ):
            doc_ids.append(doc_id)

    print(f"Keeping {len(doc_ids)} doc ids...")

    # make train and eval split
    Random(args.seed).shuffle(doc_ids)

    train_ids = doc_ids[: args.train_size]
    eval_ids = doc_ids[args.train_size : args.train_size + args.eval_size]

    # save IDs
    train_ids_path = os.path.join(
        args.data_dir, f"{medline.lang_from}_{medline.lang_to}_train_ids.txt"
    )
    with open(train_ids_path, "w") as f:
        f.write("\n".join(train_ids))

    eval_ids_path = os.path.join(
        args.data_dir, f"{medline.lang_from}_{medline.lang_to}_eval_ids.txt"
    )
    with open(eval_ids_path, "w") as f:
        f.write("\n".join(eval_ids))
