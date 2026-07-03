import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel
from .refiner import RefinerPromptProcessor

class _DiCoRBase(nn.Module):
    def __init__(self, backbone, classifier, refineHead, args):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.refineHead = refineHead
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None

    def _encode_text(self, text: torch.Tensor, l_mask: torch.Tensor):
        l_feats = self.text_encoder(text, attention_mask=l_mask.squeeze(-1))[0]
        l_feats = l_feats.permute(0, 2, 1)
        return l_feats

    def _run_backbone(self,
                      x: torch.Tensor,
                      text: torch.Tensor,
                      l_feats: torch.Tensor,
                      l_mask: torch.Tensor):
        out_backbone = self.backbone(
            x,
            l_feats,
            l_mask.permute(0, 2, 1),
            input_ids=text,
        )
        x_c1, x_c2, x_c3, x_c4 = out_backbone['features']
        return out_backbone, x_c1, x_c2, x_c3, x_c4

    def _coarse_logits(self, x_c1, x_c2, x_c3, x_c4, input_shape):
        coarse_logits_120 = self.classifier(x_c4, x_c3, x_c2, x_c1)
        coarse_logits_480 = F.interpolate(coarse_logits_120, size=input_shape, mode='bilinear', align_corners=True)
        return coarse_logits_120, coarse_logits_480

    @staticmethod
    def _dilate01(x01: torch.Tensor, r: int):
        if r <= 0:
            return x01
        k = 2 * r + 1
        return F.max_pool2d(x01, kernel_size=k, stride=1, padding=r)

    @staticmethod
    def _erode01(x01: torch.Tensor, r: int):
        if r <= 0:
            return x01
        return 1.0 - _DiCoRBase._dilate01(1.0 - x01, r)


class DiCoRCoarse(_DiCoRBase):
    def forward(self,
                x: torch.Tensor,
                text: torch.Tensor,
                l_mask: torch.Tensor):
        input_shape = x.shape[-2:]
        l_feats = self._encode_text(text, l_mask)
        out_backbone, x_c1, x_c2, x_c3, x_c4 = self._run_backbone(
            x, text, l_feats, l_mask
        )
        pre_feats = out_backbone.get('features_pre_vmsf', [None, None, None, None])
        coarse_logits_120, coarse_logits_480 = self._coarse_logits(x_c1, x_c2, x_c3, x_c4, input_shape)
        l_star = out_backbone.get('l_feats', l_feats).permute(0, 2, 1).contiguous()
        result = {
            'x': coarse_logits_480,
            'coarse_logits_120': coarse_logits_120,
            'l_star': l_star,
            'l_mask': l_mask,
            'x_c1_star': x_c1,
            'x_c2_star': x_c2,
            'x_c3_star': x_c3,
            'x_c4_star': x_c4,
            'x_pre_c1_star': pre_feats[0],
            'x_pre_c2_star': pre_feats[1],
            'x_pre_c3_star': pre_feats[2],
            'x_pre_c4_star': pre_feats[3],
        }
        localization_out = out_backbone.get('localization_guidance')
        if localization_out is not None:
            result.update(localization_out)
        return result


class DiCoRRefinerTrain(_DiCoRBase):
    PROMPT_PROCESSOR = RefinerPromptProcessor()

    def __init__(self, backbone=None, classifier=None, refineHead=None, args=None):
        if backbone is None and classifier is None:
            nn.Module.__init__(self)
            if refineHead is None:
                raise ValueError("DiCoRRefinerTrain offline mode requires refineHead")
            self.backbone = None
            self.classifier = None
            self.text_encoder = None
            self.refineHead = refineHead
        else:
            super().__init__(backbone, classifier, refineHead, args)

    def forward(self, x: torch.Tensor, prompt_override: torch.Tensor):

        input_shape = x.shape[-2:]
        prompt = prompt_override
        if prompt.shape[-2:] != input_shape:
            prompt = F.interpolate(prompt, size=input_shape, mode="bilinear", align_corners=False)
        prompt = prompt.detach()

        coarse_logits_480 = self.PROMPT_PROCESSOR.logits_from_prob_fg(prompt)
        focus_map = self.PROMPT_PROCESSOR.build_focus_map(prompt)

        refine_in = torch.cat([x, prompt], dim=1)
        delta_logits_480 = self.refineHead(refine_in)
        final_logits_480 = coarse_logits_480 + delta_logits_480

        return {
            "x": final_logits_480,
            "coarse_logits_480": coarse_logits_480,
            "delta_logits_480": delta_logits_480,
            "focus_map": focus_map,
        }


class DiCoRRefinerTest(_DiCoRBase):
    def forward(self,
                x: torch.Tensor,
                text: torch.Tensor,
                l_mask: torch.Tensor):
        input_shape = x.shape[-2:]
        l_feats = self._encode_text(text, l_mask)
        out_backbone, x_c1, x_c2, x_c3, x_c4 = self._run_backbone(
            x, text, l_feats, l_mask
        )
        pre_feats = out_backbone.get('features_pre_vmsf', [None, None, None, None])
        coarse_logits_120, coarse_logits_480 = self._coarse_logits(x_c1, x_c2, x_c3, x_c4, input_shape)

        prompt = torch.softmax(coarse_logits_480, dim=1)[:, 1:2].detach()
        refine_in = torch.cat([x, prompt], dim=1)
        delta_logits_480 = self.refineHead(refine_in)
        final_logits_480 = coarse_logits_480 + delta_logits_480  

        result = {
            'x': final_logits_480,
            'coarse_logits_120': coarse_logits_120,
            'coarse_logits_480': coarse_logits_480,
            'delta_logits_480': delta_logits_480,
            'l_star': out_backbone.get('l_feats', l_feats).permute(0, 2, 1).contiguous(),
            'l_mask': l_mask,
            'x_c1_star': x_c1,
            'x_c2_star': x_c2,
            'x_c3_star': x_c3,
            'x_c4_star': x_c4,
            'x_pre_c1_star': pre_feats[0],
            'x_pre_c2_star': pre_feats[1],
            'x_pre_c3_star': pre_feats[2],
            'x_pre_c4_star': pre_feats[3],
            'router_reg': out_backbone.get('router_reg', None),
        }
        localization_out = out_backbone.get('localization_guidance')
        if localization_out is not None:
            result.update(localization_out)
        return result
