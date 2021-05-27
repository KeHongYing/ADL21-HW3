import json
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class NLGDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: AutoTokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return {
            "labels": self.tokenizer.encode(instance["title"]),
            "input_ids": self.tokenizer.encode(instance["maintext"]),
        }


if __name__ == "__main__":
    with open("cache/train.json", "r") as f:
        data = json.load(f)

    dataset = NLGDataset(data)
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=32,
    )

    for d in dataloader:
        print(d["labels"])
        print(d["input_ids"])
        print()

        break
