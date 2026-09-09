import torch
from .mask_predictor import SimpleDecoding
from .refiner import RefineUNet
from .backbone import MultiModalSwinTransformer
from ._utils import DiCoRCoarse, DiCoRRefinerTrain, DiCoRRefinerTest
from .localization_guidance import build_localization_guidance, load_localization_guidance
import os

__all__ = ['dicor', 'dicor_coarse', 'dicor_refiner_train', 'dicor_refiner_test']

# return embed_dim, depths, num_heads
def _swin_hyper_by_type(swin_type):
    if swin_type == 'tiny':
        return 96, [2, 2, 6, 2], [3, 6, 12, 24]
    elif swin_type == 'small':
        return 96, [2, 2, 18, 2], [3, 6, 12, 24]
    elif swin_type == 'base':
        return 128, [2, 2, 18, 2], [4, 8, 16, 32]
    elif swin_type == 'large':
        return 192, [2, 2, 18, 2], [6, 12, 24, 48]
    else:
        raise ValueError(f"Unknown swin_type: {swin_type}")

def _window_size(pretrained, args):
    if ('window12' in (pretrained or '')) or getattr(args, 'window12', False):
        print('Window size 12!')
        return 12
    return 7


def _attach_localization_guidance(backbone, cfg=None):
    if not getattr(cfg, 'use_localization_guidance', False):
        return
    if cfg.locate_ckpt:
        guidance = load_localization_guidance(
            ckpt_path=cfg.locate_ckpt,
            alpha=cfg.alpha,
            lambda_geo=cfg.lambda_geo,
            map_location='cpu',
        )
        source = cfg.locate_ckpt
    else:
        guidance = build_localization_guidance(alpha=cfg.alpha, lambda_geo=cfg.lambda_geo)
        source = 'random initialization'
    backbone.set_localization_guidance(guidance)
    print(f"Attached LocalizationGuidanceModule from {source}")

def _build_dicor_components(pretrained, pretrained_refineHead, args, cfg=None, with_refiner=False):
    embed_dim, depths, num_heads = _swin_hyper_by_type(args.swin_type)
    window_size = _window_size(pretrained, args)
    out_indices = (0, 1, 2, 3)
    backbone = MultiModalSwinTransformer(
        embed_dim=embed_dim,
        depths=depths,
        swin_num_heads=num_heads,
        window_size=window_size,
        num_vmsf_blocks=args.num_vmsf_blocks,
        num_heads_fusion=args.num_heads_fusion,
        out_indices=out_indices,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False,
        visual_fusion=cfg.coarse.visual_fusion,
    )

    if pretrained:
        print('Initializing Multi-modal Swin Transformer weights from ' + pretrained)
        backbone.init_weights(pretrained=pretrained)
    else:
        print('Randomly initialize Multi-modal Swin Transformer weights.')
        backbone.init_weights()
    _attach_localization_guidance(backbone, cfg=cfg)

    classifier = SimpleDecoding(8 * embed_dim)
    if with_refiner:
        mask_ch = 1 # softmax frontend prob
        in_ch = 3 + mask_ch
        base_ch = 64
        refineHead = RefineUNet(in_ch=in_ch, base_ch=base_ch)
    else:
        refineHead = None   

    # load pretrained refine head when configured
    if with_refiner and refineHead is not None:
        ckpt_path = pretrained_refineHead
        if ckpt_path and os.path.isfile(ckpt_path):
            sd = torch.load(ckpt_path, map_location="cpu")

            if isinstance(sd, dict) and len(sd) > 0 and list(sd.keys())[0].startswith('refineHead.'):
                new_sd = {k.replace('refineHead.', ''): v for k, v in sd.items() if k.startswith('refineHead.')}
                refineHead.load_state_dict(new_sd, strict=False)
                print(f"Loaded pretrained refine head (extracted from full/prefixed dict) from {ckpt_path}")
            else:
                refineHead.load_state_dict(sd, strict=False)
                print(f"Loaded pretrained refine head from {ckpt_path}")
        elif ckpt_path:
            print(f"[WARN] Pretrained refine head path not found: {ckpt_path}")

    return backbone, classifier, refineHead


def dicor_coarse(pretrained='', pretrained_refineHead='', args=None, cfg=None):
    backbone, classifier, _ = _build_dicor_components(
        pretrained,
        pretrained_refineHead,
        args,
        cfg=cfg,
        with_refiner=False,
    )
    return DiCoRCoarse(backbone, classifier, None, args)


def dicor_refiner_train(pretrained='', pretrained_refineHead='', args=None, cfg=None):
    backbone, classifier, refineHead = _build_dicor_components(
        pretrained,
        pretrained_refineHead,
        args,
        cfg=cfg,
        with_refiner=True,
    )
    return DiCoRRefinerTrain(backbone, classifier, refineHead, args)


def dicor_refiner_test(pretrained='', pretrained_refineHead='', args=None, cfg=None):
    backbone, classifier, refineHead = _build_dicor_components(
        pretrained,
        pretrained_refineHead,
        args,
        cfg=cfg,
        with_refiner=True,
    )
    return DiCoRRefinerTest(backbone, classifier, refineHead, args)


def dicor(pretrained='', pretrained_refineHead='', args=None, cfg=None):
    return dicor_coarse(pretrained, pretrained_refineHead, args, cfg=cfg)
