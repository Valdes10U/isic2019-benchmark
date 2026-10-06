import timm


def build_model(name, num_classes=8):
    return timm.create_model(name, pretrained=True, num_classes=num_classes)


def swin_reshape_transform(x):
    # en timm, la salida de norm en Swin es (B, H, W, C); Grad-CAM espera (B, C, H, W)
    return x.permute(0, 3, 1, 2)


def gradcam_target_layer(model, name):
    """Devuelve (capa objetivo, reshape_transform) según el protocolo."""
    if name == "resnet50":
        return model.layer4[-1], None
    if name == "efficientnet_b0":
        return model.conv_head, None
    if name == "swin_tiny_patch4_window7_224":
        return model.norm, swin_reshape_transform
    raise ValueError(f"Modelo sin capa de Grad-CAM definida: {name}")
