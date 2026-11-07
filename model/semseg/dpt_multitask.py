import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.dinov2 import DINOv2
from model.util.blocks import FeatureFusionBlock, _make_scratch


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, use_bn=False,
                 out_channels=(256, 512, 1024, 1024)):
        super().__init__()

        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels, out_channel, kernel_size=1, stride=1, padding=0)
            for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])

        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features, nclass, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        return self.scratch.output_conv(path_1)


class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim=0, dropout=0.0):
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, num_classes),
            )

    def forward(self, feat_map):
        return self.classifier(feat_map)


class SimpleDetectionHead(nn.Module):
    def __init__(self, in_channels, num_det_classes, hidden_dim=256):
        super().__init__()
        self.tower = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Conv2d(hidden_dim, num_det_classes, 3, padding=1)
        self.box = nn.Conv2d(hidden_dim, 4, 3, padding=1)
        self.obj = nn.Conv2d(hidden_dim, 1, 3, padding=1)

    def forward(self, feat_map):
        x = self.tower(feat_map)
        return {
            "pred_logits": self.cls(x),
            "pred_boxes": self.box(x),
            "pred_obj": self.obj(x),
        }


class MultiTaskDPT(nn.Module):
    def __init__(self, encoder_size='base', nclass=21, features=128,
                 out_channels=(96, 192, 384, 768), use_bn=False,
                 num_cls_classes=None, num_det_classes=None):
        super().__init__()
        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11],
            'large': [4, 11, 17, 23],
            'giant': [9, 19, 29, 39],
        }
        self.encoder_size = encoder_size
        self.backbone = DINOv2(model_name=encoder_size)
        self.embed_dim = self.backbone.embed_dim

        self.heads = nn.ModuleDict()
        self.heads['seg'] = DPTHead(nclass, self.embed_dim, features, use_bn, out_channels=out_channels)
        if num_cls_classes is not None and num_cls_classes > 0:
            self.heads['cls'] = ClassificationHead(self.embed_dim, num_cls_classes)
        if num_det_classes is not None and num_det_classes > 0:
            self.heads['det'] = SimpleDetectionHead(self.embed_dim, num_det_classes)

        # Keep aliases so your old code using model.head still works.
        self.head = self.heads['seg']
        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def extract_features(self, x):
        patch_h = x.shape[-2] // 14
        patch_w = x.shape[-1] // 14
        features = self.backbone.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder_size]
        )
        last = features[-1]
        b, n, d = last.shape
        feat_map = last.permute(0, 2, 1).reshape(b, d, patch_h, patch_w)
        return features, feat_map, patch_h, patch_w

    def forward_seg(self, x, comp_drop=False):
        out_features, _, patch_h, patch_w = self.extract_features(x)

        if comp_drop:
            bs, dim = out_features[0].shape[0], out_features[0].shape[-1]
            dropout_mask1 = self.binomial.sample((bs // 2, dim)).to(x.device) * 2.0
            dropout_mask2 = 2.0 - dropout_mask1
            dropout_prob = 0.5
            num_kept = int(bs // 2 * (1 - dropout_prob))
            kept_idx = torch.randperm(bs // 2, device=x.device)[:num_kept]
            dropout_mask1[kept_idx, :] = 1.0
            dropout_mask2[kept_idx, :] = 1.0
            dropout_mask = torch.cat((dropout_mask1, dropout_mask2))
            out_features = [f * dropout_mask.unsqueeze(1) for f in out_features]

        out = self.heads['seg'](out_features, patch_h, patch_w)
        out = F.interpolate(out, (patch_h * 14, patch_w * 14), mode='bilinear', align_corners=True)
        return out

    def forward(self, x, task='seg', return_features=False, comp_drop=False):
        if task == 'seg':
            logits = self.forward_seg(x, comp_drop=comp_drop)
            if return_features:
                _, feat_map, _, _ = self.extract_features(x)
                return logits, feat_map
            return logits

        out_features, feat_map, patch_h, patch_w = self.extract_features(x)
        outputs = {}
        if task in ('cls', 'all') and 'cls' in self.heads:
            outputs['cls'] = self.heads['cls'](feat_map)
        if task in ('det', 'all') and 'det' in self.heads:
            outputs['det'] = self.heads['det'](feat_map)
        if task in ('all',) and 'seg' in self.heads:
            seg = self.heads['seg'](out_features, patch_h, patch_w)
            outputs['seg'] = F.interpolate(seg, (patch_h * 14, patch_w * 14), mode='bilinear', align_corners=True)

        if task == 'cls':
            return outputs['cls']
        if task == 'det':
            return outputs['det']
        if return_features:
            outputs['features'] = feat_map
        return outputs


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def extract_backbone_state_dict(model):
    return unwrap_model(model).backbone.state_dict()


def load_backbone_state_dict(model, state_dict, strict=True):
    unwrap_model(model).backbone.load_state_dict(state_dict, strict=strict)
