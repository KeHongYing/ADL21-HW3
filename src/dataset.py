import json
from typing import List, Dict

import torch
from torch._C import dtype
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class NLGDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        input_truncation_len: int = 512,
        output_truncation_len: int = 64,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.input_truncation_len = input_truncation_len
        self.output_truncation_len = output_truncation_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    def collate_fn(self, sample) -> Dict:
        labels = []
        input_ids = []
        for s in sample:
            labels.append(
                self.tokenizer.encode(
                    s["title"],
                    max_length=self.output_truncation_len,
                    truncation=True,
                    padding="max_length",
                )
            )
            maintext = ""
            for text in s["maintext"]:
                if len(maintext + text) < self.input_truncation_len:
                    maintext += text
                else:
                    break
            input_ids.append(
                self.tokenizer.encode(
                    maintext,
                    max_length=self.input_truncation_len,
                    truncation=True,
                    padding="max_length",
                )
            )

        return {
            "labels": torch.tensor(labels, dtype=torch.long),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
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
        print(d["labels"])
        print(d["input_ids"])
        print()

        break
