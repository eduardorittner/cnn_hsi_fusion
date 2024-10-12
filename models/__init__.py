import torch


def model_generator(method: str, pretrained_model: str | None = None):
    if method == "mst_plus_plus":
        model = None
        print("'mst_plus_plus' not defined yet!")
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
