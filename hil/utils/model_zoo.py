import numpy as np
import timm
import torch
from timm.models import register_model
from timm.models.helpers import build_model_with_cfg
from timm.models.xcit import checkpoint_filter_fn, default_cfgs
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast


def _create_xcit(variant, pretrained=True, default_cfg=None, att_vis=False, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    model = build_model_with_cfg(
        XCiTCheckpoint, variant, pretrained, default_cfg=default_cfg, pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)

    return model


class XCiTCheckpoint(timm.models.xcit.XCiT):
    def __init__(self, checkpoints=None, **kwargs):
        super().__init__(**kwargs)
        self.checkpoints = checkpoints
        if checkpoints:
            if len(self.blocks) == checkpoints:
                checkpoints = list(range(0, len(self.blocks)))
            else:
                checkpoints = list(np.linspace(0, len(self.blocks) - 1, checkpoints + 2))[1:-1]
            self.checkpoints = [int(round(c)) for c in checkpoints]

    @autocast('cuda')
    def forward_features(self, x: torch.Tensor):
        if self.training:
            x.requires_grad = True
        B = x.shape[0]
        # x is (B, N, C). (Hp, Hw) is (height in units of patches, width in units of patches)
        x, (Hp, Wp) = self.patch_embed(x)

        if self.use_pos_embed:
            # `pos_embed` (B, C, Hp, Wp), reshape -> (B, C, N), permute -> (B, N, C)
            pos_encoding = self.pos_embed(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding

        x = self.pos_drop(x)
        for i_blk, blk in enumerate(self.blocks):
            if self.checkpoints and self.training:
                if i_blk in self.checkpoints:
                    x = checkpoint(blk, x, Hp, Wp)
                else:
                    x = blk(x, Hp, Wp)
            else:
                x = blk(x, Hp, Wp)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.cls_attn_blocks:
            x = blk(x)

        x = self.norm(x)[:, 0]
        return x


@register_model
def xcit_gc_nano_12_p8_224_dist(pretrained=True, **kwargs):
    model_kwargs = dict(
        patch_size=8, embed_dim=128, depth=12, num_heads=4, eta=1.0, tokens_norm=False, **kwargs)
    model = _create_xcit('xcit_nano_12_p8_224_dist', pretrained=pretrained, **model_kwargs)
    return model
