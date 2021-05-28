import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM
from tqdm import tqdm
import jsonlines

from dataset import NLGDataset
from utils import environment_set


def main(args):
    environment_set(args.seed)

    with open(args.tokenizer, "rb") as f:
        tokenizer = pickle.load(f)

    with open(args.test_file, "r") as f:
        data = json.load(f)

    dataset = NLGDataset(data, tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(args.ckpt_path).to(args.device)
    model.eval()

    result = []

    with torch.set_grad_enabled(False):
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description("[predict]")

            for data in tepoch:
                input_ids = data["input_ids"].to(args.device)
                Id = data["id"]

                pred = model.generate(input_ids, max_length=128)
                text = tokenizer.batch_decode(pred, skip_special_tokens=True)

                result.extend(
                    [{"title": title, "id": idx} for idx, title in zip(Id, text)]
                )

    with open(args.pred_file, "w") as f:
        jsonlines.Writer(f).write_all(result)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file", type=Path, help="Path to the test file.", required=True
    )
    parser.add_argument(
        "--ckpt_path", type=Path, help="Path to model checkpoint.", required=True
    )
    parser.add_argument("--pred_file", type=Path, default="output.jsonl")

    # data loader
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--seed", type=int, default=0xB06902074)
    parser.add_argument("--tokenizer", help="tokenizer path", required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
