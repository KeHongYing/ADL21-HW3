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

        decoded_preds = [self.fill_empty(pred) for pred in decoded_preds]
        decoded_labels = [self.fill_empty(label) for label in decoded_labels]

        score = get_rouge(decoded_preds, decoded_labels)
        for key in score:
            score[key] = score[key]["f"]
        
        return score

    def fill_empty(self, candidate):
        if len(candidate) > 0:
            return candidate
        return "空"
