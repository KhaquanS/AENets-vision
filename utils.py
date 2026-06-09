from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Any

import torch
import torch.nn as nn

FeatureKind = Literal["map", "tokens", "vector"]
BackboneFamily = Literal["resnet", "vgg", "vit"]


@dataclass(frozen=True)
class FeatureSpec:
    kind: FeatureKind
    channels: int
    height: Optional[int] = None
    width: Optional[int] = None
    tokens: Optional[int] = None
    has_cls_token: bool = False

    @property
    def spatial_tokens(self) -> Optional[int]:
        if self.height is None or self.width is None:
            return self.tokens
        return self.height * self.width


@dataclass(frozen=True)
class BackboneSlice:
    module: nn.Module
    family: BackboneFamily
    point: str
    input_spec: Optional[FeatureSpec] = None
    output_spec: Optional[FeatureSpec] = None


def infer_backbone_family(model: nn.Module) -> BackboneFamily:
    if hasattr(model, "conv_proj") and hasattr(model, "encoder"):
        return "vit"
    if hasattr(model, "features") and hasattr(model, "classifier"):
        return "vgg"
    if hasattr(model, "layer1") and hasattr(model, "conv1"):
        return "resnet"
    raise ValueError(f"Unsupported backbone type: {type(model).__name__}")


def build_torchvision_model(
    name: str,
    *,
    num_classes: int,
    weights: Optional[Any] = None,
    pretrained: bool = False,
    strict_head: bool = False,
) -> nn.Module:
    """
    Build a torchvision model by canonical name.

    ``weights`` can be a torchvision weight enum instance or the string "DEFAULT".
    """
    import torchvision.models as tvm
    if hasattr(tvm, "get_model"):
        if weights == "DEFAULT":
            try:
                weights = tvm.get_model_weights(name).DEFAULT
            except Exception:
                weights = None
        if weights is None and pretrained:
            try:
                weights = tvm.get_model_weights(name).DEFAULT
            except Exception:
                weights = None
        model = tvm.get_model(name, weights=weights) if weights is not None else tvm.get_model(name, weights=None)
    else:
        factory = getattr(tvm, name)
        model = factory(weights=weights if weights is not None else None)

    if strict_head:
        return model

    # Replace classification head to match the requested class count.
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            for idx in range(len(model.classifier) - 1, -1, -1):
                if isinstance(model.classifier[idx], nn.Linear):
                    model.classifier[idx] = nn.Linear(model.classifier[idx].in_features, num_classes)
                    break
        elif isinstance(model.classifier, nn.Linear):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif hasattr(model, "heads") and hasattr(model.heads, "head") and isinstance(model.heads.head, nn.Linear):
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    return model


def default_cut(family: BackboneFamily) -> str:
    if family == "resnet":
        return "layer3"
    if family == "vgg":
        return "features[23]"
    if family == "vit":
        return "encoder.layers[7]"
    raise ValueError(f"Unknown family: {family}")


def _parse_point(point: str) -> Tuple[str, Optional[int]]:
    """
    Supports strings like ``features[10]`` or ``encoder.layers[4]``.
    """
    if "[" in point and point.endswith("]"):
        base = point[: point.index("[")]
        idx = int(point[point.index("[") + 1 : -1])
        return base, idx
    return point, None


def _resnet_stage_out_channels(stage: str) -> int:
    return {"stem": 64, "layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512, "avgpool": 512, "flatten": 512, "fc": 512}[stage]


def _resnet_stage_in_channels(stage: str) -> int:
    return {"layer1": 64, "layer2": 64, "layer3": 128, "layer4": 256, "avgpool": 512, "flatten": 512, "fc": 512}.get(stage, 64)


class ResNetPrefix(nn.Module):
    def __init__(self, backbone: nn.Module, cut: str):
        super().__init__()
        self.backbone = backbone
        self.cut = cut
        self.base, self.idx = _parse_point(cut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.backbone
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)

        if self.base == "stem":
            return x

        for stage_name in ["layer1", "layer2", "layer3", "layer4"]:
            stage = getattr(m, stage_name)
            if stage_name == self.base:
                if self.idx is None:
                    x = stage(x)
                else:
                    for i, block in enumerate(stage):
                        x = block(x)
                        if i == self.idx:
                            break
                return x
            x = stage(x)

        if self.base == "avgpool":
            return m.avgpool(x)
        if self.base == "flatten":
            return torch.flatten(m.avgpool(x), 1)
        if self.base == "fc":
            x = torch.flatten(m.avgpool(x), 1)
            return m.fc(x)

        raise ValueError(f"Unsupported ResNet cut: {self.cut}")


class ResNetSuffix(nn.Module):
    def __init__(self, backbone: nn.Module, start: str):
        super().__init__()
        self.backbone = backbone
        self.start = start
        self.base, self.idx = _parse_point(start)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.backbone

        stage_order = ["layer1", "layer2", "layer3", "layer4"]
        if self.base in stage_order:
            for stage_name in stage_order[stage_order.index(self.base) :]:
                stage = getattr(m, stage_name)
                if stage_name == self.base and self.idx is not None:
                    # Start from a particular block inside a stage.
                    for i in range(self.idx, len(stage)):
                        x = stage[i](x)
                else:
                    x = stage(x)
            x = m.avgpool(x)
            x = torch.flatten(x, 1)
            return m.fc(x)

        if self.base == "avgpool":
            x = m.avgpool(x)
            x = torch.flatten(x, 1)
            return m.fc(x)

        if self.base == "flatten":
            x = torch.flatten(x, 1)
            return m.fc(x)

        if self.base == "fc":
            return m.fc(x)

        raise ValueError(f"Unsupported ResNet start: {self.start}")


def _vgg_feature_channels(backbone: nn.Module, idx: int) -> int:
    channels = 3
    for i, layer in enumerate(backbone.features):
        if isinstance(layer, nn.Conv2d):
            channels = layer.out_channels
        if i == idx:
            return channels
    return channels


def _vgg_input_channels_for_start(backbone: nn.Module, point: str) -> int:
    base, idx = _parse_point(point)
    if base == "features":
        if idx is None or idx == 0:
            return 3
        return _vgg_feature_channels(backbone, idx - 1)
    if base == "classifier":
        return backbone.classifier[0].in_features if isinstance(backbone.classifier, nn.Sequential) and isinstance(backbone.classifier[0], nn.Linear) else 25088
    if base in {"avgpool", "flatten"}:
        return _vgg_feature_channels(backbone, len(backbone.features) - 1)
    if base == "fc":
        return _vgg_feature_channels(backbone, len(backbone.features) - 1)
    raise ValueError(f"Unsupported VGG start: {point}")


class VGGPrefix(nn.Module):
    def __init__(self, backbone: nn.Module, cut: str):
        super().__init__()
        self.backbone = backbone
        self.cut = cut
        self.base, self.idx = _parse_point(cut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.backbone
        if self.base == "features":
            for i, layer in enumerate(m.features):
                x = layer(x)
                if self.idx is not None and i == self.idx:
                    break
            return x
        if self.base == "avgpool":
            x = m.features(x)
            return m.avgpool(x)
        if self.base == "flatten":
            x = m.features(x)
            x = m.avgpool(x)
            return torch.flatten(x, 1)
        if self.base == "classifier":
            x = m.features(x)
            x = m.avgpool(x)
            x = torch.flatten(x, 1)
            for i, layer in enumerate(m.classifier):
                x = layer(x)
                if self.idx is not None and i == self.idx:
                    break
            return x
        raise ValueError(f"Unsupported VGG cut: {self.cut}")


class VGGSuffix(nn.Module):
    def __init__(self, backbone: nn.Module, start: str):
        super().__init__()
        self.backbone = backbone
        self.start = start
        self.base, self.idx = _parse_point(start)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.backbone
        if self.base == "features":
            for i, layer in enumerate(m.features):
                if self.idx is not None and i < self.idx:
                    continue
                x = layer(x)
            x = m.avgpool(x)
            x = torch.flatten(x, 1)
            return m.classifier(x)
        if self.base == "avgpool":
            x = m.avgpool(x)
            x = torch.flatten(x, 1)
            return m.classifier(x)
        if self.base == "flatten":
            x = torch.flatten(x, 1)
            return m.classifier(x)
        if self.base == "classifier":
            x = m.features(x)
            x = m.avgpool(x)
            x = torch.flatten(x, 1)
            for i, layer in enumerate(m.classifier):
                if self.idx is not None and i < self.idx:
                    continue
                x = layer(x)
            return x
        raise ValueError(f"Unsupported VGG start: {self.start}")


class ViTPrefix(nn.Module):
    def __init__(self, backbone: nn.Module, cut: str):
        super().__init__()
        self.backbone = backbone
        self.cut = cut
        self.base, self.idx = _parse_point(cut)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        m = self.backbone
        x = m._process_input(x)
        n = x.shape[0]
        cls = m.class_token.expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + m.encoder.pos_embedding
        return m.encoder.dropout(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.backbone
        x = self._embed(x)
        if self.base == "encoder":
            return m.encoder.ln(m.encoder.layers(x))
        if self.base == "encoder.layers":
            for i, layer in enumerate(m.encoder.layers):
                x = layer(x)
                if self.idx is not None and i == self.idx:
                    break
            return x
        if self.base in {"heads", "logits"}:
            x = m.encoder.ln(m.encoder.layers(x))
            x = x[:, 0]
            return m.heads(x)
        raise ValueError(f"Unsupported ViT cut: {self.cut}")


class ViTSuffix(nn.Module):
    def __init__(self, backbone: nn.Module, start: str):
        super().__init__()
        self.backbone = backbone
        self.start = start
        self.base, self.idx = _parse_point(start)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.backbone
        if self.base == "encoder":
            x = m.encoder.ln(m.encoder.layers(x))
            x = x[:, 0]
            return m.heads(x)
        if self.base == "encoder.layers":
            for i, layer in enumerate(m.encoder.layers):
                if self.idx is not None and i < self.idx:
                    continue
                x = layer(x)
            x = m.encoder.ln(x)
            x = x[:, 0]
            return m.heads(x)
        if self.base in {"heads", "logits"}:
            x = m.encoder.ln(x)
            x = x[:, 0]
            return m.heads(x)
        raise ValueError(f"Unsupported ViT start: {self.start}")


def _probe_module_output_spec(module: nn.Module, dummy: torch.Tensor) -> FeatureSpec:
    """Run a dummy forward pass and convert the output tensor shape to a FeatureSpec."""
    training = module.training
    module.eval()
    with torch.no_grad():
        out = module(dummy)
    module.train(training)
    if out.dim() == 4:
        return FeatureSpec("map", channels=out.shape[1], height=out.shape[2], width=out.shape[3])
    if out.dim() == 3:
        return FeatureSpec("tokens", channels=out.shape[2], tokens=out.shape[1], has_cls_token=True)
    return FeatureSpec("vector", channels=out.shape[1])


def _make_dummy_image(backbone: nn.Module, batch_size: int = 2) -> torch.Tensor:
    """Create a minimal dummy input image on the same device as the backbone."""
    device = next(backbone.parameters()).device
    return torch.zeros(batch_size, 3, 224, 224, device=device)


def _prior_resnet_cut(start: str) -> Optional[str]:
    """
    Return the cut point whose output equals the input tensor of ``start``.
    E.g. the input to layer2 is the output of layer1, so return "layer1".
    Returns None when start is the earliest possible point (stem).
    """
    order = ["stem", "layer1", "layer2", "layer3", "layer4", "avgpool", "flatten", "fc"]
    base, idx = _parse_point(start)
    if base not in order:
        return None
    pos = order.index(base)
    if pos == 0:
        return None
    # Block-level granularity: input to block[idx] == output of block[idx-1].
    if idx is not None and idx > 0:
        return f"{base}[{idx - 1}]"
    return order[pos - 1]


def _prior_vgg_cut(backbone: nn.Module, start: str) -> Optional[str]:
    """Return the VGG cut point just before ``start``."""
    base, idx = _parse_point(start)
    if base == "features":
        if idx is None or idx == 0:
            return None
        return f"features[{idx - 1}]"
    if base == "avgpool":
        return f"features[{len(backbone.features) - 1}]"
    if base == "flatten":
        return "avgpool"
    if base == "classifier":
        return "flatten" if (idx is None or idx == 0) else f"classifier[{idx - 1}]"
    return None


def _prior_vit_cut(backbone: nn.Module, start: str) -> Optional[str]:
    """Return the ViT cut point just before ``start``."""
    base, idx = _parse_point(start)
    if base == "encoder.layers":
        if idx is None or idx == 0:
            return None
        return f"encoder.layers[{idx - 1}]"
    if base in {"encoder", "heads", "logits"}:
        n = len(backbone.encoder.layers)
        return f"encoder.layers[{n - 1}]"
    return None


def teacher_prefix_slice(backbone: nn.Module, cut: Optional[str] = None) -> BackboneSlice:
    family = infer_backbone_family(backbone)
    cut = cut or default_cut(family)

    if family == "resnet":
        module = ResNetPrefix(backbone, cut)
    elif family == "vgg":
        module = VGGPrefix(backbone, cut)
    elif family == "vit":
        module = ViTPrefix(backbone, cut)
    else:
        raise ValueError(f"Unsupported backbone family: {family}")

    output_spec = _probe_module_output_spec(module, _make_dummy_image(backbone))
    return BackboneSlice(module=module, family=family, point=cut, output_spec=output_spec)


def student_suffix_slice(backbone: nn.Module, start: Optional[str] = None) -> BackboneSlice:
    family = infer_backbone_family(backbone)
    start = start or default_cut(family)

    if family == "resnet":
        module = ResNetSuffix(backbone, start)
        output_spec = FeatureSpec("vector", channels=backbone.fc.out_features)
        prior_cut = _prior_resnet_cut(start)
    elif family == "vgg":
        module = VGGSuffix(backbone, start)
        output_spec = FeatureSpec("vector", channels=backbone.classifier[-1].out_features if isinstance(backbone.classifier, nn.Sequential) and isinstance(backbone.classifier[-1], nn.Linear) else 1000)
        prior_cut = _prior_vgg_cut(backbone, start)
    elif family == "vit":
        module = ViTSuffix(backbone, start)
        output_spec = FeatureSpec("vector", channels=getattr(backbone.heads.head, "out_features", 1000))
        prior_cut = _prior_vit_cut(backbone, start)
    else:
        raise ValueError(f"Unsupported backbone family: {family}")

    # Probe the tensor shape that *enters* the suffix by running a prefix that stops
    # one stage before `start`.  This is what the adapter encoder must output.
    dummy = _make_dummy_image(backbone)
    if prior_cut is not None:
        prefix_cls = {"resnet": ResNetPrefix, "vgg": VGGPrefix, "vit": ViTPrefix}[family]
        prior_prefix = prefix_cls(backbone, prior_cut)
        input_spec = _probe_module_output_spec(prior_prefix, dummy)
    else:
        # start is the very first stage; input is the raw image tensor.
        input_spec = FeatureSpec("map", channels=dummy.shape[1], height=dummy.shape[2], width=dummy.shape[3])

    return BackboneSlice(module=module, family=family, point=start, input_spec=input_spec, output_spec=output_spec)


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    params = module.parameters() if not trainable_only else (p for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in params)


def freeze_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad = False
    module.eval()
    return module


def describe_feature_spec(spec: FeatureSpec) -> str:
    if spec.kind == "map":
        return f"map[{spec.channels}, h={spec.height or '?'}, w={spec.width or '?'}]"
    if spec.kind == "tokens":
        return f"tokens[{spec.tokens or '?'}, d={spec.channels}]"
    return f"vector[{spec.channels}]"