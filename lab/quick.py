import torch
import yaml
from vocos.pretrained import instantiate_class


aud = torch.rand(1, 22050)
config_path = "./matcha/configs/vocos-matcha.yaml"
# config_path = "./configs/vocos-std.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
params = config["model"]["init_args"]
feat = instantiate_class(args=(), init=params["feature_extractor"])
back = instantiate_class(args=(), init=params["backbone"])
head = instantiate_class(args=(), init=params["head"])
mel = feat(aud)
mel = mel.unsqueeze(0)
h = back(mel)
ret = head(h)


"""

class VM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.back = back
        self.head = head

    def forward(self, mel):
        h = self.back(mel)
        audio = self.head(h)
        return audio


model = VM()
model = model.to("cpu").eval()

dummy_input = torch.rand(1, model.back.input_channels, 64)
dynamic_axes = {
    "mels": {0: "batch_size", 2: "time"},
    "audio": {0: "batch_size", 1: "time"},
}
torch.onnx.export(
   model=model,
   args=dummy_input,
   f="vocos.c.onnx",
   input_names=["mels"],
   output_names=["audio"],
   dynamic_axes=dynamic_axes,
    opset_version=16,
     export_params=True,
   do_constant_folding=True,
)


options = torch.onnx.ExportOptions(dynamic_shapes=True)
exported_model = torch.onnx.dynamo_export(model, dummy_input, export_options=options)
exported_model.save("vm.onnx")
"""