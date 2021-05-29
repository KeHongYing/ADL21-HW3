import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)

from transformers.integrations import TensorBoardCallback

from trainer_dataset import NLGDataset
from ComputeMatrics import ComputeMatrics
from utils import environment_set

TRAIN = "train"
DEV = "val"
SPLITS = [TRAIN, DEV]


def main(args):
    environment_set(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    data_paths = {split: args.cache_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, NLGDataset] = {
        split: NLGDataset(
            split_data, tokenizer, args.input_truncation_len, args.output_truncation_len
        )
        for split, split_data in data.items()
    }

    model = AutoModelForSeq2SeqLM.from_pretrained(args.backbone)

    backbone = (
        args.backbone if "/" not in args.backbone else args.backbone.split("/")[1]
    )
    ckpt_dir = args.ckpt_dir / backbone
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(ckpt_dir / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    train_args = Seq2SeqTrainingArguments(
        str(ckpt_dir / args.model),
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=10,
        num_train_epochs=args.num_epoch,
        predict_with_generate=True,
        fp16=True,
        gradient_accumulation_steps=args.accumulate_steps,
        adafactor=True,
        seed=args.seed % 2147483647,
        load_best_model_at_end=True,
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
        default="./trainer_ckpt/",
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
    parser.add_argument("--batch_size", type=int, default=8)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--accumulate_steps", type=int, default=8)

    # misc
    parser.add_argument("--seed", type=int, default=(0xB06902074))

    # model
    parser.add_argument(
        "--backbone",
        help="bert backbone",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--input_truncation_len", help="maintext truncate length", type=int, default=512
    )
    parser.add_argument(
        "--output_truncation_len", help="title truncate length", type=int, default=64
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
