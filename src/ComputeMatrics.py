from transformers import AutoTokenizer
from tw_rouge import get_rouge
import numpy as np


class ComputeMatrics:
    def __init__(self, tokenizer: AutoTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # # Rouge expects a newline after each sentence
        # decoded_preds = [
        #     "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
        # ]
        # decoded_labels = [
        #     "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
        # ]

        # result = self.metric.compute(
        #     predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        # )

        result = get_rough(decoded_preds, decoded_labels)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # Add mean generated length
        prediction_lens = [
            np.count_nonzero(pred != self.tokenizer.pad_token_id)
            for pred in predictions
        ]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}
