"""
Image encoder backends for ActionToVerbTransformer.

All encoders share the same interface:
    - Input:  (B, 3, H, W) float tensor
    - forward():        (B, num_tokens, d_model) — projected embeddings
    - get_embeddings():  (B, num_tokens, raw_dim) — raw backbone features (no proj)
    - Attribute: self.num_tokens  (int)

Available encoders (select via --image_encoder):
    scratch   -- ViT-style Conv2d patch embedding, trained from scratch (default)
    resnet18  -- ImageNet-pretrained ResNet-18, backbone frozen
    dinov2    -- DINOv2 ViT-S/14 (timm), backbone frozen, patch tokens pooled to 64
    dinov2_s  -- DINOv2 ViT-S/14 spatially pooled to 7×7 = 49 tokens
    dinov2_b  -- DINOv2 ViT-B/14 spatially pooled to 7×7 = 49 tokens
    vc1       -- VC-1 ViT-B/16 spatially pooled to 7×7 = 49 tokens
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
        assert img_size % patch_size == 0, (
            f"img_size {img_size} must be divisible by patch_size {patch_size}"
        )
        self.num_tokens = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, 3, H, W) -> (B, num_tokens, d_model)
        return self.proj(x).flatten(2).transpose(1, 2)

    def get_embeddings(self, x):
        # Scratch encoder has no separate backbone; proj IS the embedding
        return self.forward(x)


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
        self.embed_dim = 512
        self.proj = nn.Linear(512, d_model)

    def _predict(self, x):
        """Raw backbone features: (B, 49, 512)."""
        with torch.no_grad():
            feat = self.features(x)  # (B, 512, 7, 7)
        return feat.flatten(2).transpose(1, 2)  # (B, 49, 512)

    def forward(self, x):
        return self.proj(self._predict(x))  # (B, 49, d_model)

    def get_embeddings(self, x):
        """Return raw backbone features without projection."""
        return self._predict(x)  # (B, 49, 512)


# 3a. DINOv2 ViT-S/14 (timm, frozen, patch tokens pooled to 64 — "dinov2" generic)
#     Use "dinov2_s", "dinov2_b", or "vc1" for the spatially-pooled 7×7 = 49-token variants.


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
        self.backbone = timm.create_model("vit_small_patch14_dinov2", pretrained=True)
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.num_tokens = 64
        dino_dim = self.backbone.embed_dim  # 384 for ViT-S
        self.embed_dim = dino_dim
        self.pool = nn.AdaptiveAvgPool1d(self.num_tokens)
        self.proj = nn.Linear(dino_dim, d_model)

        # Detect expected input size from model (newer timm uses 518)
        _img = getattr(
            self.backbone,
            "img_size",
            getattr(self.backbone.patch_embed, "img_size", (224, 224)),
        )
        self._img_size = (_img, _img) if isinstance(_img, int) else tuple(_img)

    def _predict(self, x):
        """Raw pooled patch tokens: (B, 64, 384)."""
        x = F.interpolate(x, size=self._img_size, mode="bilinear", align_corners=False)
        with torch.no_grad():
            patch_tokens = self.backbone.get_intermediate_layers(x, n=1)[0]
        pooled = self.pool(patch_tokens.transpose(1, 2)).transpose(1, 2)
        return pooled  # (B, 64, 384)

    def forward(self, x):
        return self.proj(self._predict(x))  # (B, 64, d_model)

    def get_embeddings(self, x):
        """Return raw backbone features without projection."""
        return self._predict(x)  # (B, 64, 384)


# 3b. ViT patch-pool encoder: DINOv2-S, DINOv2-B, VC-1 (7×7 spatial pool → 49 tokens)


class ViTPatchPoolEncoder(nn.Module):
    """Frozen ViT patch encoder with spatial pooling to pool_size×pool_size tokens.

    Supports:
        dinov2_s  — DINOv2 ViT-S/14, embed_dim=384
        dinov2_b  — DINOv2 ViT-B/14, embed_dim=768
        vc1       — VC-1 ViT-B/16 (facebook/vc1-base), embed_dim=768
    """

    def __init__(self, variant: str, d_model: int = 128, pool_size: int = 7):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required. Install with: pip install timm")

        self.pool_size = pool_size
        self.variant = variant

        if variant == "dinov2_s":
            self.vit = timm.create_model(
                "vit_small_patch14_dinov2", pretrained=True, num_classes=0, img_size=224
            )
            embed_dim = 384
            self.grid_size = 16  # 224 / 14
        elif variant == "dinov2_b":
            self.vit = timm.create_model(
                "vit_base_patch14_dinov2", pretrained=True, num_classes=0, img_size=224
            )
            embed_dim = 768
            self.grid_size = 16
        elif variant == "vc1":
            self.vit = timm.create_model(
                "vit_base_patch16_224", pretrained=False, num_classes=0
            )
            embed_dim = 768
            self.grid_size = 14  # 224 / 16
            self._load_vc1_weights()
        else:
            raise ValueError(
                f"Unknown ViT variant '{variant}'. Choose from: dinov2_s, dinov2_b, vc1"
            )

        self.vit.eval()
        for p in self.vit.parameters():
            p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.proj = nn.Linear(embed_dim, d_model)
        self.embed_dim = embed_dim
        self.num_tokens = pool_size * pool_size

    def _load_vc1_weights(self):
        from huggingface_hub import hf_hub_download

        path = hf_hub_download("facebook/vc1-base", "pytorch_model.bin")
        vc1_state = torch.load(path, map_location="cpu", weights_only=True)["model"]
        missing, unexpected = self.vit.load_state_dict(vc1_state, strict=False)
        n_loaded = len(vc1_state) - len(unexpected)
        print(
            f"VC-1: loaded {n_loaded} params, {len(missing)} missing, "
            f"{len(unexpected)} unexpected"
        )

    def _predict(self, x):
        """Raw spatially-pooled patch tokens: (B, pool_size^2, embed_dim)."""
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.no_grad():
            features = self.vit.forward_features(x)  # (B, 1+N, embed_dim)
        patches = features[:, 1:, :]  # drop CLS
        B, N, D = patches.shape
        h = w = self.grid_size
        patches = patches.transpose(1, 2).reshape(B, D, h, w)
        patches = self.pool(patches)  # (B, D, pool_size, pool_size)
        return patches.flatten(2).transpose(1, 2)  # (B, pool_size^2, D)

    def forward(self, x):
        return self.proj(self._predict(x))  # (B, num_tokens, d_model)

    def get_embeddings(self, x):
        """Return raw backbone features without projection."""
        return self._predict(x)  # (B, num_tokens, embed_dim)


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
        self.embed_dim = 2048
        self.proj = nn.Linear(2048, d_model)

    def _predict(self, x):
        """Raw R3M features: (B, 49, 2048) — global vector repeated to 49 tokens."""
        x_uint8 = (x * 255.0).clamp(0, 255)
        with torch.no_grad():
            feat = self.backbone(x_uint8)  # (B, 2048)
        return feat.unsqueeze(1).expand(-1, self.num_tokens, -1)  # (B, 49, 2048)

    def forward(self, x):
        return self.proj(self._predict(x))  # (B, 49, d_model)

    def get_embeddings(self, x):
        """Return raw backbone features without projection."""
        return self._predict(x)  # (B, 49, 2048)


# Builder function to select encoder by name


def build_image_encoder(
    name: str, d_model: int = 128, img_size: int = 200, patch_size: int = 25
) -> nn.Module:
    """Return the requested image encoder.

    Args:
        name:       "scratch" | "resnet18" | "dinov2" | "r3m"
                    | "dinov2_s" | "dinov2_b" | "vc1"
                    | "patch" (alias for scratch)
        d_model:    output embedding dimension
        img_size:   input image size (used only by ScratchPatchEmbed)
        patch_size: patch size (used only by ScratchPatchEmbed)
    """
    name = name.lower()
    if name in ("scratch", "patch"):
        return ScratchPatchEmbed(
            img_size=img_size, patch_size=patch_size, d_model=d_model
        )
    elif name == "resnet18":
        return ResNet18Encoder(d_model=d_model)
    elif name == "dinov2":
        return DINOv2Encoder(d_model=d_model)
    elif name in ("dinov2_s", "dinov2_b", "vc1"):
        return ViTPatchPoolEncoder(variant=name, d_model=d_model)
    elif name == "r3m":
        return R3MEncoder(d_model=d_model)
    else:
        raise ValueError(
            f"Unknown image encoder '{name}'. "
            f"Choose from: scratch, resnet18, dinov2, dinov2_s, dinov2_b, vc1, r3m"
        )
