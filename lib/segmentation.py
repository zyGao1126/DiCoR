import torch
from .mask_predictor import SimpleDecoding
from .refiner import RefineUNet
from .backbone import MultiModalSwinTransformerV2
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


def _localization_guidance_config(cfg):
    joint_cfg = getattr(cfg, 'joint_tune', None)
    if joint_cfg is not None:
        return joint_cfg if getattr(joint_cfg, 'enabled', False) else None
    return cfg if getattr(cfg, 'use_localization_guidance', False) else None

def _attach_localization_guidance(backbone, cfg=None):
    locate_cfg = _localization_guidance_config(cfg)
    if locate_cfg is None:
        return
    if getattr(locate_cfg, 'locate_ckpt', ''):
        guidance, payload = load_localization_guidance(
            ckpt_path=locate_cfg.locate_ckpt,
            alpha=locate_cfg.alpha,
            map_location='cpu',
        )
        source = locate_cfg.locate_ckpt
    else:
        guidance, payload = build_localization_guidance(alpha=locate_cfg.alpha)
        source = 'random initialization'
    backbone.set_localization_guidance(guidance)
    print(
        f"Attached LocalizationGuidanceModule from {source} "
        f"(feature_key={payload.get('feature_key', '')}, "
        f"alpha={locate_cfg.alpha:.3f})"
    )

def _build_dicor_components(pretrained, pretrained_refineHead, args, cfg=None, with_refiner=False):
    embed_dim, depths, num_heads = _swin_hyper_by_type(args.swin_type)
    window_size = _window_size(pretrained, args)
    out_indices = (0, 1, 2, 3)
    backbone = MultiModalSwinTransformerV2(
        embed_dim=embed_dim,
        depths=depths,
        swin_num_heads=num_heads,
        window_size=window_size,
        num_tmem=args.num_tmem,
        num_heads_fusion=args.num_heads_fusion,
        out_indices=out_indices,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False,
        use_lvmsf=cfg.coarse.use_lvmsf,
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
