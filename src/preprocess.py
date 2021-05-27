import json
import random
import logging
import pickle
from typing import List, Dict, Tuple
from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm.auto import tqdm
from transformers import AutoTokenizer

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def train_val_split(
    data: List[Dict], test_size: float = 0.2, shuffle: bool = True
) -> Tuple[List[Dict], List[Dict]]:
    if shuffle:
        random.shuffle(data)

    train = data[int(test_size * len(data)) :]
    val = data[: int(test_size * len(data))]

    return train, val


def main(args):
    random.seed(args.rand_seed)

    with open(args.data_dir / args.data, "r") as f:
        data = [json.loads(line) for line in f.read().splitlines()]

    if args.tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    else:
        tokenizer = pickle.load(open(args.tokenizer, "rb"))

    output = []
    for d in tqdm(data, desc="preprocessing data..."):
        title = d["title"]
        maintext = d["maintext"]
        Id = d["id"]

        split_text = maintext.split("\n")
        output.append(
            {
                "title": title,
                "maintext": split_text,
                "id": Id,
            }
        )

    if args.training:
        train, val = train_val_split(output, test_size=0.1)

        with open(args.output_dir / "train.json", "w") as f:
            json.dump(train, f, ensure_ascii=False, indent=4)
        with open(args.output_dir / "val.json", "w") as f:
            json.dump(val, f, ensure_ascii=False, indent=4)
    else:
        with open(args.output_dir / args.data, "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to the question.",
        default="train.jsonl",
    )
    parser.add_argument("--rand_seed", type=int, help="Random seed.", default=13)
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the processed file.",
        default="./cache/",
    )
    parser.add_argument(
        "--training", help="preprocess training data or not", action="store_true"
    )
    parser.add_argument("--max_len", type=int, help="token max length.", default=512)
    parser.add_argument(
        "--backbone",
        help="tokenizer backbone",
        type=str,
        default="google/mt5-small",
    )
    # tokenizer
    parser.add_argument(
        "--tokenizer_dir",
        type=Path,
        help="Directory to save the tokenizer.",
        default="./tokenizer",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="tokenizer path.",
        default=None,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
