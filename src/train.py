import json
import pickle
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import DefaultDict, Dict

from tqdm import tqdm
import torch
import tensorflow as tf
from torch.utils.data import DataLoader

from transformers import AutoModelForSeq2SeqLM, Adafactor, AutoTokenizer

from dataset import NLGDataset
from ComputeMatrics import ComputeMatrics
from choose_low_utility_gpu import choose_low_utility_gpu

TRAIN = "train"
DEV = "val"
SPLITS = [TRAIN, DEV]
learning_curve = DefaultDict(list)


def iter_loop(
    dataloader: DataLoader,
    model: AutoModelForSeq2SeqLM,
    optimizer: torch.optim,
    device: torch.device,
    mode: str,
    compute_matric: ComputeMatrics,
) -> None:
    total_correct = 0
    total_rouge = DefaultDict(int)
    total_loss = 0

    if mode == TRAIN:
        model.train()
    elif mode == DEV:
        model.eval()

    with torch.set_grad_enabled(mode == TRAIN):
        with tqdm(dataloader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"[{mode:>5}]")

                input_ids = data["input_ids"].to(device)
                labels = data["labels"].to(device)

                pred = model(input_ids, labels=labels)

                loss = pred.loss
                output = torch.argmax(pred.logits, dim=-1)
                if mode == DEV:
                    rouge_score = compute_matric((output, labels))
                    for rouge in rouge_score:
                        total_rouge[rouge] += rouge_score[rouge]["f"]
                        learning_curve[rouge].append(rouge_score[rouge]["f"])

                correct = ((output == labels).type(torch.float)).mean().item()
                total_correct += correct
                total_loss += loss

                if mode == TRAIN:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if mode == DEV:
                    tepoch.set_postfix(
                        loss=f"{loss.item():>.4f}",
                        Acc=f"{correct:>.4f}",
                        Rouge1=f"{rouge_score['rouge-1']['f']:>.4f}",
                        Rouge2=f"{rouge_score['rouge-2']['f']:>.4f}",
                        RougeL=f"{rouge_score['rouge-l']['f']:>.4f}",
                    )
                else:
                    tepoch.set_postfix(
                        loss=f"{loss.item():>.4f}",
                        Acc=f"{correct:>.4f}",
                    )

            total_correct /= len(tepoch)
            total_loss /= len(tepoch)
            if mode == DEV:
                for rouge in total_rouge:
                    total_rouge[rouge] /= len(tepoch)

                print(
                    f"[{mode:>5}] ",
                    f"Acc: {total_correct:>.4f},",
                    f"loss: {total_loss:>.7f},",
                    f"Rouge1: {total_rouge['rouge-1']:>.4f}",
                    f"Rouge2: {total_rouge['rouge-2']:>.4f}",
                    f"RougeL: {total_rouge['rouge-l']:>.4f}",
                )
            else:
                print(
                    f"[{mode:>5}] ",
                    f"Acc: {total_correct:>.4f},",
                    f"loss: {total_loss:>.7f},",
                )

    return total_correct, total_loss


def environment_set(seed: int = 42, limit: int = 5000):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(choose_low_utility_gpu(limit))
    torch.manual_seed(args.seed)
    tf.random.set_seed(seed)


def main(args):
    environment_set(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    data_paths = {split: args.cache_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, NLGDataset] = {
        split: NLGDataset(split_data, tokenizer) for split, split_data in data.items()
    }

    dataloader = {
        split: DataLoader(
            datasets[split],
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=datasets[split].collate_fn,
        )
        for split in SPLITS
    }

    model = AutoModelForSeq2SeqLM.from_pretrained(args.backbone).to(args.device)

    optimizer = Adafactor(params=model.parameters(), weight_decay=args.weight_decay)

    max_acc, min_loss = 0, 100
    early_stop = 0

    backbone = (
        args.backbone if "/" not in args.backbone else args.backbone.split("/")[1]
    )
    ckpt_dir = args.ckpt_dir / backbone
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    compute_matrics = ComputeMatrics(tokenizer)
    for epoch in range(args.num_epoch):
        print(f"Epoch: {epoch + 1}")
        iter_loop(
            dataloader[TRAIN], model, optimizer, args.device, TRAIN, compute_matrics
        )
        acc, loss = iter_loop(
            dataloader[DEV], model, optimizer, args.device, DEV, compute_matrics
        )

        if loss < min_loss:
            max_acc = acc
            min_loss = loss
            torch.save(
                model.state_dict(),
                ckpt_dir / f"{args.model}_best.pt",
            )
            print(f"model is better than before, save model to {args.model}_best.pt")

        if loss > min_loss:
            early_stop += 1
        else:
            early_stop = 0
            min_loss = loss

        if early_stop == 10:
            print("Early stop...")
            break

    print(f"Done! Best model Acc: {(100 * max_acc):>.4f}%")
    model.save_pretrained(ckpt_dir)

    with open(ckpt_dir / f"learning_curve.json", "w") as f:
        json.dump(learning_curve, f)

    with open("result_match.txt", "a") as f:
        f.write(f"{backbone}/{args.model}, {max_acc:>5f}, {min_loss:>.5f}\n")


def parse_args() -> Namespace:
    parser = ArgumentParser()
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
    parser.add_argument("--batch_size", type=int, default=8)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=500)

    # misc
    parser.add_argument("--seed", type=int, default=0xB06902074)

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
