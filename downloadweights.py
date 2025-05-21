import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, logging
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

class XTRLinear(torch.nn.Module):
    def __init__(self, in_features=768, out_features=128, bias=False):
        super().__init__()

    def forward(self, x):
        return self.linear(x)

class XTR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("google/xtr-base-en", use_safetensors=True).encoder
        to_dense_path = hf_hub_download(repo_id="google/xtr-base-en", filename="2_Dense/pytorch_model.bin")

        self.encoder.linear = torch.nn.Linear(768, 128, bias=False)
        state = torch.load(to_dense_path)
        other = {}
        other["weight"] = state["linear.weight"]
        self.encoder.linear.load_state_dict(other)

xtr = XTR()
save_file(xtr.state_dict(), "xtr.safetensors")
