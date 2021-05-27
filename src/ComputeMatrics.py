from transformers import AutoTokenizer
from tw_rouge import get_rouge
import torch


class ComputeMatrics:
    def __init__(self, tokenizer: AutoTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        return get_rouge(decoded_preds, decoded_labels)
