import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)

from transformers.integrations import TensorBoardCallback

from dataset import NLGDataset
from ComputeMatrics import ComputeMatrics
from choose_low_utility_gpu import choose_low_utility_gpu

TRAIN = "train"
DEV = "val"
SPLITS = [TRAIN, DEV]


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    data_paths = {split: args.cache_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, NLGDataset] = {
        split: NLGDataset(split_data, tokenizer) for split, split_data in data.items()
    }

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.backbone)

    backbone = (
        args.backbone if "/" not in args.backbone else args.backbone.split("/")[1]
    )
    ckpt_dir = args.ckpt_dir / backbone
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(choose_low_utility_gpu())

    train_args = Seq2SeqTrainingArguments(
        str(ckpt_dir / f"{args.model}.pt"),
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=10,
        num_train_epochs=args.num_epoch,
        predict_with_generate=True,
        fp16=True,
        gradient_accumulation_steps=args.accumulate_steps,
        adafactor=True,
        seed=args.seed,
        load_best_model_at_end=True,
        dataloader_num_workers=32,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    compute_metrics = ComputeMatrics(tokenizer)

    callback_list = [
        EarlyStoppingCallback(early_stopping_patience=5),
        TensorBoardCallback(),
    ]
    trainer = Seq2SeqTrainer(
        model,
        train_args,
        train_dataset=datasets[TRAIN],
        eval_dataset=datasets[DEV],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callback_list,
    )

    trainer.train()


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="model name.",
        default="model",
    )

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # data loader
    parser.add_argument("--batch_size", type=int, default=16)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=500)
    parser.add_argument("--accumulate_steps", type=int, default=8)

    # misc
    parser.add_argument("--seed", type=int, default=(0xB06902074 % (1 << 31 - 1)))

    # model
    parser.add_argument(
        "--backbone",
        help="bert backbone",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--no_pretrained", help="do not use pretrained weight", action="store_true"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
