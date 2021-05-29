import json
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
        Id = []
        for s in sample:
            labels.append(
                self.tokenizer.encode(
                    s["title"],
                    max_length=self.output_truncation_len,
                    truncation=True,
                    padding="max_length",
                )
            )
            # length = [len(s["maintext"][-1])] * len(s["maintext"])
            # for i in range(len(s["maintext"]) - 2, -1, -1):
            #     length[i] = len(s["maintext"][i]) + length[i + 1]

            maintext = ""
            for text in s["maintext"]:
                if len(maintext + text) < self.input_truncation_len:
                    maintext += text
                else:
                    break
            # for idx, text in enumerate(s["maintext"]):
            #     if length[idx] < self.input_truncation_len / 2:
            #         maintext += text

            text_with_pos = [[idx, text] for idx, text in enumerate(s["maintext"])]
            random.shuffle(text_with_pos)
            text = []
            length = 0
            for idx, text in text_with_pos:
                if length + len(text) <= self.input_truncation_len:
                    text.append([idx, text])
                    length += len(text)
                else:
                    break

            text.sort(key=lambda x: x[0])
            maintext = "".join([t for _, t in text])

            input_ids.append(
                self.tokenizer.encode(
                    maintext,
                    max_length=self.input_truncation_len,
                    truncation=True,
                    padding="max_length",
                )
            )

            Id.append(s["id"])

        return {
            "labels": torch.tensor(labels, dtype=torch.long),
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
