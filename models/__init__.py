import torch
from .MST import MST
from .MST_Plus_Plus import MST_Plus_Plus
from .mirnet import MIRNet


def model_generator(method: str, pretrained_model: str | None = None):
    if method == "mst":
        model = MST(dim=31, stage=2, num_blocks=[4, 7, 5])
    elif method == "mst_plus_plus":
        model = MST_Plus_Plus()
    elif method == "mirnet":
        model = MIRNet(in_channels=34)
    else:
        raise Exception(f"No method named {method}")

    if pretrained_model:
        print(f"Loading model from '{pretrained_model}'")
        checkpoint = torch.load(pretrained_model)
        # Why are we doing this?
        model.load_state_dict(
            {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()},
            strict=True,
        ),

    return model
