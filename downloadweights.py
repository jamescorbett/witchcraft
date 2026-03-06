import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import save_file
import os

from transformers import T5EncoderModel, T5PreTrainedModel, SinqConfig
from sinq.sinqlinear_hf import SINQLinear

# SINQ quantization replaces weight tensors with None; T5's _init_weights
# then crashes trying to re-initialize them.  Patch to skip those modules.
_original_init_weights = T5PreTrainedModel._init_weights
def _patched_init_weights(self, module):
    try:
        _original_init_weights(self, module)
    except AttributeError:
        pass
T5PreTrainedModel._init_weights = _patched_init_weights

snapshot_download(repo_id="google/xtr-base-en", local_dir="xtr-base-en",
                  local_dir_use_symlinks=False, revision="main")

def export_to_openvino(model, output_dir="openvino_model"):
    """Convert PyTorch model directly to OpenVINO IR (no ONNX intermediate)."""
    import openvino as ov

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    dummy_input = torch.randint(0, 32128, (1, 512), dtype=torch.long)

    print("Converting PyTorch model directly to OpenVINO IR...")
    with torch.no_grad():
        test_output = model(dummy_input)
        print(f"Model output shape: {test_output.shape} (expected: [1, 512, 128])")

        ov_model = ov.convert_model(
            model,
            example_input=dummy_input,
            input=[("input_ids", [1, -1], torch.int64)]  # Dynamic sequence length
        )

    xml_path = os.path.join(output_dir, "xtr-ov.xml")

    print("Saving FP32 base model (will be quantized to INT4 by quantize_int4.py)...")
    ov.save_model(ov_model, xml_path)

    bin_path = xml_path.replace(".xml", ".bin")
    if not os.path.exists(xml_path) or not os.path.exists(bin_path):
        raise FileNotFoundError(f"OpenVINO conversion failed - files not found: {xml_path}, {bin_path}")

    print(f"Saved OpenVINO model: {xml_path}")
    return xml_path, bin_path


class XTR(nn.Module):
    def __init__(self, quantization_config=None):
        super().__init__()

        encoder_model = T5EncoderModel.from_pretrained(
            "google/xtr-base-en",
            quantization_config=quantization_config,
            use_safetensors=True,
        )
        self.encoder = encoder_model.encoder

        self.linear = nn.Linear(768, 128, bias=False)
        to_dense_path = hf_hub_download(
            repo_id="google/xtr-base-en",
            filename="2_Dense/pytorch_model.bin",
        )
        state = torch.load(to_dense_path, map_location="cpu", weights_only=True)
        self.linear.load_state_dict({"weight": state["linear.weight"]})

    def forward(self, input_ids):
        encoder_output = self.encoder(input_ids)
        hidden_states = encoder_output.last_hidden_state if hasattr(encoder_output, 'last_hidden_state') else encoder_output[0]
        return self.linear(hidden_states)


# Quantize with SINQ then dequantize back to FP16 — SINQ's value is the
# optimized quantization grid, which survives the round-trip better than naive RTN
sinq_cfg = SinqConfig(nbits=4, group_size=64, tiling_mode="1D", method="sinq")
xtr = XTR(quantization_config=sinq_cfg)

# Replace every SINQLinear with a standard nn.Linear holding the dequantized weight.
# Keep weights in FP32 for inference/OpenVINO export; save FP16 to safetensors.
for name, module in list(xtr.named_modules()):
    if not isinstance(module, SINQLinear):
        continue
    w = module.dequantize().float()
    replacement = nn.Linear(w.shape[1], w.shape[0], bias=module.bias is not None)
    replacement.weight = nn.Parameter(w)
    if module.bias is not None:
        replacement.bias = nn.Parameter(module.bias.float())
    parts = name.split(".")
    parent = xtr
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], replacement)

fp16_state_dict = {k: v.half().cpu() for k, v in xtr.state_dict().items()}
save_file(fp16_state_dict, "xtr.safetensors")
print(f"Saved xtr.safetensors with {len(fp16_state_dict)} tensors")

import shutil
shutil.copy("xtr-base-en/config.json", "assets/config.json")
shutil.copy("xtr-base-en/tokenizer.json", "assets/tokenizer.json")

try:
    model_xml_path, model_bin_path = export_to_openvino(xtr)
    print("\n" + "="*60)
    print("OpenVINO FP32 base model export successful!")
    print("="*60)

    model_size = os.path.getsize(model_bin_path) / (1024 * 1024)
    print(f"FP32 base model size: {model_size:.2f} MB")
    print(f"Model files: {model_xml_path}, {model_bin_path}")

    print("\nTo create INT4 quantized model (~66 MB), run:")
    print("  python quantize-int4.py")
    print("\n" + "="*60)

except Exception as e:
    print(f"\nERROR: OpenVINO export failed: {e}")
    print(f"\nMake sure you have OpenVINO and NNCF installed:")
    print(f"  pip install openvino nncf")
    import traceback
    traceback.print_exc()
    exit(1)
