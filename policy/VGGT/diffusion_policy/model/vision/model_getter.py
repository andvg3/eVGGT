import torch
import torchvision

from vggt.model.vggt_ldm import VGGT_LDM

def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    # resnet_new = torch.nn.Sequential(
    #     resnet,
    #     torch.nn.Linear(512, 128)
    # )
    # return resnet_new
    return resnet


def get_vggt(name, weights=None, **kwargs):
    # init model
    vggt = VGGT_LDM(latent_pretraining=False, **kwargs)

    # load weights if provided
    if weights is not None:
        checkpoint = torch.load(weights, map_location="cpu")
        
        # if saved with DataParallel or contains 'module.' prefix
        state_dict = checkpoint.get("state_dict", checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v
        
        missing, unexpected = vggt.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"Missing keys when loading weights: {missing}")
        if unexpected:
            print(f"Unexpected keys in checkpoint: {unexpected}")
        print(f"Loaded from {weights}")
    # freeze all params
    for p in vggt.parameters():
        p.requires_grad = False

    return vggt

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m

    r3m.device = "cpu"
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to("cpu")
    return resnet_model
