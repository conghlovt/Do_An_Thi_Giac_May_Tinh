import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Basic residual block cho ResNet tự xây dựng.

    Nhánh chính: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm.
    Nhánh shortcut: identity nếu shape khớp, hoặc Conv 1x1 nếu cần đổi kênh/stride.
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Residual connection bắt buộc của BasicBlock.
        out = F.relu(out + shortcut)
        return out


class CustomResNet(nn.Module):
    """ResNet-18 mini tự code từ đầu, phù hợp ảnh khuôn mặt 64x64."""

    def __init__(self, block, layers, num_classes=8, in_channels=3, dropout=0.3):
        super().__init__()

        self.in_channels = 64

        # Stem nhỏ hơn ResNet chuẩn để hợp với ảnh nhỏ 64x64.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """Tạo một stage gồm nhiều BasicBlock liên tiếp."""
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []

        for block_stride in strides:
            blocks.append(block(self.in_channels, out_channels, block_stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def custom_resnet18(num_classes, in_channels=3):
    """ResNet-18 tự xây dựng từ BasicBlock với số block [2, 2, 2, 2]."""
    return CustomResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        in_channels=in_channels,
        dropout=0.3,
    )


def hsemotion_efficientnet_b0(num_classes):
    """Dựng backbone giống checkpoint enet_b0_8_best_vgaf đã fine-tune."""
    try:
        from hsemotion.facial_emotions import HSEmotionRecognizer
    except ImportError as error:
        raise ImportError(
            "Checkpoint EfficientNet cần hsemotion. Hãy cài: pip install hsemotion==0.3.0"
        ) from error

    # hsemotion 0.3.0 load model timm hoàn chỉnh từ cache. File này là
    # model tin cậy của thư viện; PyTorch mới cần weights_only=False.
    original_torch_load = torch.load

    def trusted_hsemotion_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = trusted_hsemotion_load
    try:
        recognizer = HSEmotionRecognizer(
            model_name="enet_b0_8_best_vgaf", device="cpu"
        )
    finally:
        torch.load = original_torch_load

    model = recognizer.model
    feature_dim = int(recognizer.classifier_weights.shape[1])
    model.classifier = nn.Linear(feature_dim, num_classes)

    # Tương thích state_dict giữa checkpoint timm cũ và timm hiện tại.
    for module in model.modules():
        if module.__class__.__name__ in {"DepthwiseSeparableConv", "InvertedResidual"}:
            if not hasattr(module, "conv_s2d"):
                module.conv_s2d = None
            if not hasattr(module, "aa"):
                module.aa = nn.Identity()

    return model


def build_model(num_classes, architecture="custom_resnet"):
    """Dựng đúng kiến trúc theo metadata lưu trong checkpoint."""
    if architecture == "hsemotion_enet_b0_8_best_vgaf":
        return hsemotion_efficientnet_b0(num_classes=num_classes)
    if architecture in {"custom_resnet", "custom_resnet18", None}:
        return custom_resnet18(num_classes=num_classes, in_channels=3)
    raise ValueError(f"Kiến trúc checkpoint chưa được hỗ trợ: {architecture}")
