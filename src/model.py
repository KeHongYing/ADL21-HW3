import pickle
from typing import Dict

import torch

from transformers import AutoModel, AutoConfig


class NLG(torch.nn.Module):
    def __init__(self, model_name: str = None, config: str = None) -> None:
        super(NLG, self).__init__()
        if model_name is not None:
            self.backbone = AutoModel.from_pretrained(model_name)
        else:
            self.backbone = AutoModel.from_config(
                pickle.load(open(config, "rb")) if config is not None else AutoConfig()
            )

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        x = self.backbone(input_ids=batch)["last_hidden_state"]

        return {"output": x}
