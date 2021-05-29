import json
from os import truncate
import random
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class NLGDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        input_truncation_len: int = 512,
        output_truncation_len: int = 64,
        mode: str = "train",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.input_truncation_len = input_truncation_len
        self.output_truncation_len = output_truncation_len
        self.mode = mode

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        input_ids = ""
        for text in instance["maintext"]:
            if len(input_ids) + len(text) <= self.input_truncation_len:
                input_ids += text

        return (
            {
                "input_ids": self.tokenizer.encode(
                    input_ids,
                    truncation=True,
                    max_length=self.input_truncation_len,
                    padding="max_length",
                ),
                "labels": self.tokenizer.encode(
                    instance["title"],
                    truncation=True,
                    max_length=self.output_truncation_len,
                    padding="max_length",
                ),
            }
            if self.mode == "train"
            else {
                "input_ids": self.tokenizer.encode(
                    input_ids,
                    truncation=True,
                    max_length=self.input_truncation_len,
                    padding="max_length",
                ),
                "id": instance["id"],
            }
        )

    def collate_fn(self, sample: Dict) -> Dict:
        input_ids = [s["input_ids"] for s in sample]
        Id = [s["id"] for s in sample]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "id": Id,
        }


if __name__ == "__main__":
    with open("cache/train.json", "r") as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    dataset = NLGDataset(data, tokenizer)
    dataloader = DataLoader(
        dataset, shuffle=True, batch_size=32, collate_fn=dataset.collate_fn
    )

    for d in dataloader:
        # print(d["labels"])
        # print(d["input_ids"])
        print(d["input_ids"].shape)
