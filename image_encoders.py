"""
Image encoder backends for ActionToVerbTransformer.

All encoders share the same interface:
    - Input:  (B, 3, H, W) float tensor
    - Output: (B, num_tokens, d_model) float tensor
    - Attribute: self.num_tokens  (int)

Available encoders (select via --image_encoder):
    scratch   -- ViT-style Conv2d patch embedding, trained from scratch (default)
    resnet18  -- ImageNet-pretrained ResNet-18, backbone frozen
    dinov2    -- DINOv2 ViT-S/14 (timm), backbone frozen, patch tokens pooled to 64
    r3m       -- R3M ResNet-50 (robotics-specific), backbone frozen, global feat repeated to 49 tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. Scratch patch embedding (existing baseline, now behind common interface)

class ScratchPatchEmbed(nn.Module):
    """
    ViT-style patch embedding learned from scratch.
    """

    def __init__(self, img_size=200, patch_size=25, in_channels=3, d_model=128):
        super().__init__()
        assert img_size % patch_size == 0, \
            f"img_size {img_size} must be divisible by patch_size {patch_size}"
        self.num_tokens = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, d_model,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 3, H, W) -> (B, num_tokens, d_model)
        return self.proj(x).flatten(2).transpose(1, 2)


# 2. ResNet-18 (ImageNet pretrained, frozen)

class ResNet18Encoder(nn.Module):
    """
    ImageNet-pretrained ResNet-18 feature extractor.
    """

    def __init__(self, d_model=128):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Drop avgpool and fc — keep everything up through layer4
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        for p in self.features.parameters():
            p.requires_grad = False

        # 7x7 spatial grid = 49 tokens, each with 512 channels
        self.num_tokens = 49
        self.proj = nn.Linear(512, d_model)

    def forward(self, x):
        # x: (B, 3, H, W)
        with torch.no_grad():
            feat = self.features(x)          # (B, 512, 7, 7)
        feat = feat.flatten(2).transpose(1, 2)  # (B, 49, 512)
        return self.proj(feat)               # (B, 49, d_model)


# 3. DINOv2 ViT-S/14 (timm, frozen, patch tokens pooled to 64)

class DINOv2Encoder(nn.Module):
    """
    DINOv2 ViT-S/14 patch-token encoder.
    Requires: pip install timm
    """

    def __init__(self, d_model=128):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for DINOv2Encoder. Install with: pip install timm"
            )

        # vit_small_patch14_dinov2 — lightweight, strong embodied features
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2", pretrained=True
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.num_tokens = 64
        dino_dim = self.backbone.embed_dim   # 384 for ViT-S
        self.pool = nn.AdaptiveAvgPool1d(self.num_tokens)
        self.proj = nn.Linear(dino_dim, d_model)

        # DINOv2 expects 224x224 — register as buffer so it moves with .to(device)
        self.register_buffer(
            "_resize_size", torch.tensor([224, 224]), persistent=False
        )

    def forward(self, x):
        # Resize to 224x224 (CALVIN images are 200x200)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        with torch.no_grad():
            # get_intermediate_layers returns patch tokens only (no CLS)
            patch_tokens = self.backbone.get_intermediate_layers(x, n=1)[0]
            # patch_tokens: (B, 256, 384)

        # Pool 256 → 64 tokens: pool operates on last dim, so transpose
        pooled = self.pool(patch_tokens.transpose(1, 2)).transpose(1, 2)
        # pooled: (B, 64, 384)
        return self.proj(pooled)  # (B, 64, d_model)


# 4. R3M (robotics-specific, frozen)

class R3MEncoder(nn.Module):
    """
    R3M ResNet-50 encoder pretrained on human video for robot manipulation.
    Requires: pip install r3m
    See: https://github.com/facebookresearch/r3m
    """

    def __init__(self, d_model=128):
        super().__init__()
        try:
            from r3m import load_r3m
        except ImportError:
            raise ImportError(
                "r3m is required for R3MEncoder. "
                "Install with: pip install r3m  "
                "(see https://github.com/facebookresearch/r3m)"
            )

        r3m = load_r3m("resnet50")
        r3m.eval()
        self.backbone = r3m

        for p in self.backbone.parameters():
            p.requires_grad = False

        # R3M outputs a 2048-dim global vector; repeat to 49 tokens
        self.num_tokens = 49
        self.proj = nn.Linear(2048, d_model)

    def forward(self, x):
        x_uint8 = (x * 255.0).clamp(0, 255)

        with torch.no_grad():
            feat = self.backbone(x_uint8)    # (B, 2048)

        feat = feat.unsqueeze(1).expand(-1, self.num_tokens, -1)  # (B, 49, 2048)
        return self.proj(feat)               # (B, 49, d_model)


# Builder function to select encoder by name

def build_image_encoder(name: str, d_model: int = 128,
                         img_size: int = 200, patch_size: int = 25) -> nn.Module:
    """Return the requested image encoder.

    Args:
        name:       "scratch" | "resnet18" | "dinov2" | "r3m"
        d_model:    output embedding dimension
        img_size:   input image size (used only by ScratchPatchEmbed)
        patch_size: patch size (used only by ScratchPatchEmbed)
    """
    name = name.lower()
    if name == "scratch":
        return ScratchPatchEmbed(img_size=img_size, patch_size=patch_size,
                                 d_model=d_model)
    elif name == "resnet18":
        return ResNet18Encoder(d_model=d_model)
    elif name == "dinov2":
        return DINOv2Encoder(d_model=d_model)
    elif name == "r3m":
        return R3MEncoder(d_model=d_model)
    else:
        raise ValueError(
            f"Unknown image encoder '{name}'. "
            f"Choose from: scratch, resnet18, dinov2, r3m"
        )
