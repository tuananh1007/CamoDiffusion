

import logging
import numpy as np
import operator
from collections import OrderedDict
from typing import Any, Mapping
import diffdist.functional as diff_dist
import torch
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils import comm
from detectron2.utils.memory import retry_if_cuda_oom
from mask2former.maskformer_model import MaskFormer
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import (
    MLP,
    MultiScaleMaskedTransformerDecoder,
)
from torch import nn
from torch.nn import functional as F

from DiffCIS.data.build import prompt_labels, get_instance_labels

from .clip import ClipAdapter, MaskCLIP, build_clip_text_embed
from .helper import ensemble_logits_with_labels

logger = logging.getLogger(__name__)


# Ref:https://stackoverflow.com/questions/27049998/convert-a-mixed-nested-list-to-a-nested-tuple
def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)


@torch.no_grad()
def _concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if comm.get_world_size() == 1:
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def get_world_batch_sizes(batch_size: int, device):
    batch_size = torch.as_tensor([batch_size], dtype=torch.long, device=device)
    global_batch_sizes = _concat_all_gather(batch_size)
    return global_batch_sizes


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors, with dynamic batch size.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if comm.get_world_size() == 1:
        return tensor
    global_batch_sizes = get_world_batch_sizes(tensor.shape[0], tensor.device)
    max_batch_size = global_batch_sizes.max().item()
    padded_tensor = torch.zeros(
        max_batch_size, *tensor.shape[1:], device=tensor.device, dtype=tensor.dtype
    )
    padded_tensor[: tensor.shape[0]] = tensor

    tensors_gather = [
        torch.ones((max_batch_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        for _ in range(comm.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, padded_tensor, async_op=False)

    results = []
    for i, batch_size in enumerate(global_batch_sizes):
        results.append(tensors_gather[i][:batch_size])

    output = torch.cat(results, dim=0)
    return output


def dist_collect(tensor):
    """
    Performs all_gather operation on the provided tensors, with dynamic batch size.
    Use diff_dist to get gradient
    """
    if comm.get_world_size() == 1:
        return tensor
    global_batch_sizes = get_world_batch_sizes(tensor.shape[0], tensor.device)
    max_batch_size = global_batch_sizes.max().item()
    padded_tensor = torch.zeros(
        max_batch_size, *tensor.shape[1:], device=tensor.device, dtype=tensor.dtype
    )
    padded_tensor[: tensor.shape[0]] = tensor

    tensors_gather = [
        torch.ones((max_batch_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        for _ in range(comm.get_world_size())
    ]
    tensors_gather = diff_dist.all_gather(tensors_gather, padded_tensor)

    results = []
    for i, batch_size in enumerate(global_batch_sizes):
        results.append(tensors_gather[i][:batch_size])

    output = torch.cat(results, dim=0)
    return output


class DiffCIS(MaskFormer):
    def ignored_state_dict(self, destination=None, prefix=""):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        for name, module in self.named_modules(remove_duplicate=True):
            if module is self:
                continue
            if module is not None and hasattr(module, "ignored_state_dict"):
                module.ignored_state_dict(destination, prefix + name + ".")
        return destination

    def _open_state_dict(self):
        return {
            "sem_seg_head.num_classes": self.sem_seg_head.num_classes,
            "metadata": self.metadata,
            "test_topk_per_image": self.test_topk_per_image,
            "semantic_on": self.semantic_on,
            "panoptic_on": self.panoptic_on,
            "instance_on": self.instance_on
        }

    def _save_open_state_dict(self, destination, prefix):
        for k, v in self._open_state_dict().items():
            destination[prefix + k] = v

    def open_state_dict(self, destination=None, prefix=""):
        if destination is None:
            destination = OrderedDict()
        self._save_open_state_dict(destination, prefix)
        for name, module in self.named_modules(remove_duplicate=True):
            if module is self:
                continue
            if module is not None and hasattr(module, "open_state_dict"):
                module.open_state_dict(destination, prefix + name + ".")
        return destination

    def load_open_state_dict(self, state_dict: Mapping[str, Any]):
        for k, v in state_dict.items():
            # handle nested modules
            if len(k.rsplit(".", 1)) == 2:
                prefix, suffix = k.rsplit(".", 1)
                operator.attrgetter(prefix)(self).__setattr__(suffix, v)
            else:
                self.__setattr__(k, v)
            assert operator.attrgetter(k)(self) == v, f"{k} is not loaded correctly"


class CategoryDiffCIS(DiffCIS):
    def __init__(
        self,
        *,
        category_head=None,
        clip_head=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.category_head = category_head
        self.clip_head = clip_head

    def cal_pred_logits(self, outputs):
        # [B, Q, C]
        mask_embed = outputs["mask_embed"]
        # [K, C]
        text_embed = outputs["text_embed"]
        # [1, C]
        text_embed = outputs["text_embed"]
        null_embed = outputs["null_embed"]

        labels = outputs["labels"]

        mask_embed = F.normalize(mask_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        logit_scale = outputs["logit_scale"]

        # [B, Q, K]
        pred = logit_scale * (mask_embed @ text_embed.t())

        pred = ensemble_logits_with_labels(pred, labels, ensemble_method="max")

        null_embed = F.normalize(null_embed, dim=-1)
        null_pred = logit_scale * (mask_embed @ null_embed.t())

        # [B, Q, K+1]
        pred = torch.cat([pred, null_pred], dim=-1)

        return pred

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the
                        values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        denormalized_images = ImageList.from_tensors(
            [x["image"].to(self.device) / 255.0 for x in batched_inputs]
        )

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)
        outputs["images"] = denormalized_images.tensor

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            if self.category_head is not None:
                category_head_outputs = self.category_head(outputs, targets)
                outputs.update(category_head_outputs)
                # inplace change pred_logits
                outputs["pred_logits"] = self.cal_pred_logits(outputs)
                if "aux_outputs" in outputs:
                    for aux_outputs in outputs["aux_outputs"]:
                        aux_outputs.update(category_head_outputs)
                        # inplace change pred_logits
                        aux_outputs["pred_logits"] = self.cal_pred_logits(aux_outputs)

            # CLIP head needs output to prepare targets
            # disable for now
            # targets = self.clip_head.prepare_targets(outputs, targets)

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            return losses
        else:

            # get text_embeddings
            outputs.update(self.category_head(outputs))

            outputs["pred_logits"] = self.cal_pred_logits(outputs)

            mask_pred_results = outputs["pred_masks"]
            mask_cls_results = outputs["pred_logits"]

            if self.clip_head is not None:
                if self.clip_head.with_bg:
                    # [B, Q, K+1]
                    outputs["pred_open_logits"] = outputs["pred_logits"]
                    outputs.update(self.clip_head(outputs))
                    mask_cls_results = outputs["pred_open_logits"]
                else:
                    # [B, Q, K]
                    outputs["pred_open_logits"] = outputs["pred_logits"][..., :-1]
                    outputs.update(self.clip_head(outputs))

                    # merge with bg scores
                    open_logits = outputs["pred_open_logits"]

                    # in case the prediction is not binary
                    binary_probs = torch.zeros(
                        (mask_cls_results.shape[0], mask_cls_results.shape[1], 2),
                        device=mask_cls_results.device,
                        dtype=mask_cls_results.dtype,
                    )
                    binary_probs[..., -1] = F.softmax(mask_cls_results, dim=-1)[..., -1]
                    binary_probs[..., 0] = 1 - binary_probs[..., -1]

                    masks_class_probs = F.softmax(open_logits, dim=-1)
                    # [B, Q, K+1]
                    mask_cls_results = torch.cat(
                        [masks_class_probs * binary_probs[..., 0:1], binary_probs[..., 1:2]], dim=-1
                    )
                    # NOTE: mask_cls_results is already multiplied with logit_scale,
                    # avoid double scale, which cause overflow in softmax
                    # mask_cls_results = torch.log(mask_cls_results + 1e-8) * outputs["logit_scale"]
                    mask_cls_results = torch.log(mask_cls_results + 1e-8)

            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["instances"] = instance_r

            return processed_results


class DiffCISMultiScaleMaskedTransformerDecoder(MultiScaleMaskedTransformerDecoder):
    def __init__(
        self,
        *,
        class_embed=None,
        mask_embed=None,
        post_mask_embed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert self.mask_classification

        if class_embed is not None:
            self.class_embed = class_embed
        if mask_embed is not None:
            self.mask_embed = mask_embed
        if post_mask_embed is not None:
            assert mask_embed is None
        self.post_mask_embed = post_mask_embed

    def forward(self, x, mask_features, mask=None, *, inputs_dict=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(
                self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None]
            )

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []
        predictions_extra_results = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask, extra_results = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0], inputs_dict=inputs_dict
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        predictions_extra_results.append(extra_results)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index],
                query_pos=query_embed,
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed
            )

            ## FFN output of Diffusion U-Net
            output = self.transformer_ffn_layers[i](output)

            ##  



            outputs_class, outputs_mask, attn_mask, extra_results = self.forward_prediction_heads(
                output,
                mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                inputs_dict=inputs_dict,
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_extra_results.append(extra_results)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            ),
        }

        # adding extra_results to out and out["aux_outputs"]
        for k in predictions_extra_results[-1].keys():
            out[k] = predictions_extra_results[-1][k]
            for i in range(len(predictions_extra_results) - 1):
                out["aux_outputs"][i][k] = predictions_extra_results[i][k]

        return out

    def forward_prediction_heads(
        self, output, mask_features, attn_mask_target_size, *, inputs_dict=None
    ):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)

        extra_results = dict()

        mask_embed_results = self.mask_embed(decoder_output)
        if isinstance(mask_embed_results, dict):
            mask_embed = mask_embed_results.pop("mask_embed")
            extra_results.update(mask_embed_results)
        # BC
        else:
            mask_embed = mask_embed_results

        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        if self.post_mask_embed is not None:
            post_mask_embed_results = self.post_mask_embed(
                decoder_output, mask_embed, mask_features, outputs_class, outputs_mask
            )

            if "outputs_mask" in post_mask_embed_results:
                outputs_mask = post_mask_embed_results.pop("outputs_mask")

            extra_results.update(post_mask_embed_results)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(
            outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False
        )
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend,
        # while ``False`` values will be unchanged.
        attn_mask = (
            attn_mask.sigmoid()
            .flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
            < 0.5
        ).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask, extra_results


class PseudoClassEmbed(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        # predict as foreground only
        fg_logits = torch.ones((*x.shape[:-1], self.num_classes), dtype=x.dtype, device=x.device)
        bg_logits = torch.zeros((*x.shape[:-1], 1), dtype=x.dtype, device=x.device)
        logits = torch.cat([fg_logits, bg_logits], dim=-1)
        return logits


class PooledMaskEmbed(nn.Module):
    def __init__(
        self,
        hidden_dim,
        mask_dim,
        projection_dim,
        temperature=0.07,
    ):
        super().__init__()
        self.pool_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.mask_embed = nn.Sequential(
            nn.LayerNorm(mask_dim), MLP(mask_dim, hidden_dim, projection_dim, 3)
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

        self.mask_pooling = MaskPooling()

    def forward(self, decoder_output, input_mask_embed, mask_features, pred_logits, pred_masks):
        """
        Args:
            decoder_output: [B, Q, C]
            input_mask_embed: [B, Q, C]
            mask_features: [B, C, H, W]
            pred_logits: [B, Q, K+1]
            pred_masks: [B, Q, H, W]
        """
        mask_pooled_x = self.mask_pooling(mask_features, pred_masks)
        mask_pooled_results = self.mask_pooling(mask_features, pred_masks)
        mask_pooled_x = mask_pooled_results["mask_pooled_features"]
        outputs_mask = mask_pooled_results.get("outputs_mask", None)

        mask_pooled_x = self.pool_proj(mask_pooled_x)

        mask_pooled_x += decoder_output

        mask_embed = self.mask_embed(mask_pooled_x)

        logit_scale = torch.clamp(self.logit_scale.exp(), max=10)

        output = {
            "mask_embed": mask_embed,
            "mask_pooled_features": mask_pooled_x,
            "logit_scale": logit_scale,
        }

        if outputs_mask is not None:
            output["outputs_mask"] = outputs_mask

        return output






class MaskPooling(nn.Module):
    def __init__(
        self,
        hard_pooling=True,
        mask_threshold=0.5,
    ):
        super().__init__()
        # if the pooling is hard, it's not differentiable
        self.hard_pooling = hard_pooling
        self.mask_threshold = mask_threshold

    def extra_repr(self) -> str:
        return f"hard_pooling={self.hard_pooling}\n" f"mask_threshold={self.mask_threshold}\n"

    def forward(self, x, mask):
        """
        Args:
            x: [B, C, H, W]
            mask: [B, Q, H, W]
        """

        assert x.shape[-2:] == mask.shape[-2:]

        mask = mask.detach()

        mask = mask.sigmoid()

        if self.hard_pooling:
            mask = (mask > self.mask_threshold).to(mask.dtype)

        denorm = mask.sum(dim=(-1, -2), keepdim=True) + 1e-8

        mask_pooled_x = torch.einsum(
            "bchw,bqhw->bqc",
            x,
            mask / denorm,
        )

        output = {"mask_pooled_features": mask_pooled_x}

        return output


class WordEmbed(nn.Module):
    def __init__(
        self,
        projection_dim,
        clip_model_name="ViT-L-14",
        word_dropout=0.0,
        word_tags="noun_phrase",
        num_words=8,
        prompt="photo",
    ):
        super().__init__()

        self.clip_model_name = clip_model_name
        self.clip = ClipAdapter(name=self.clip_model_name, normalize=False)

        if projection_dim < 0:
            self.text_proj = nn.Identity()
        else:
            self.text_proj = nn.Linear(self.clip.dim_latent, projection_dim)

        self.test_labels = None
        self._test_text_embed_dict = OrderedDict()

        import nltk

        if comm.get_local_rank() == 0:
            nltk.download("popular", quiet=True)
            nltk.download("universal_tagset", quiet=True)
        comm.synchronize()
        self.nltk = nltk

        self.word_dropout = word_dropout
        self.word_tags = word_tags
        self.num_words = num_words
        self.prompt = prompt

    def extra_repr(self) -> str:
        return (
            f"clip_model_name={self.clip_model_name},\n"
            f"word_dropout={self.word_dropout},\n"
            f"word_tags={self.word_tags},\n"
            f"num_words={self.num_words}"
        )

    @property
    def device(self):
        return self.clip.device

    def _open_state_dict(self):
        return {"test_labels": self.test_labels}

    def _save_open_state_dict(self, destination, prefix):
        for k, v in self._open_state_dict().items():
            destination[prefix + k] = v

    def open_state_dict(self, destination=None, prefix=""):
        if destination is None:
            destination = OrderedDict()
        self._save_open_state_dict(destination, prefix)
        for name, module in self.named_modules(remove_duplicate=True):
            if module is self:
                continue
            if module is not None and hasattr(module, "open_state_dict"):
                module.open_state_dict(destination, prefix + name + ".")
        return destination

    @torch.no_grad()
    def build_text_embed(self, labels, verbose=False):
        return build_clip_text_embed(
            clip_model_name=self.clip.clip,
            labels=labels,
            verbose=verbose,
        )

    def get_and_cache_test_text_embed(self, labels):
        labels = to_tuple(labels)
        if labels not in self._test_text_embed_dict:
            text_embed = self.build_text_embed(labels, verbose=True)
            if len(self._test_text_embed_dict) > 3:
                # pop the first element, only caching 3 elements
                self._test_text_embed_dict.pop(list(self._test_text_embed_dict.keys())[0])
            self._test_text_embed_dict[labels] = text_embed.cpu()
        else:
            text_embed = self._test_text_embed_dict[labels].to(self.device)
        return text_embed

    def get_tag(self, caption, tags):
        if not isinstance(tags, (list, tuple)):
            tags = [tags]
        ret = []
        for (word, pos) in self.nltk.pos_tag(self.nltk.word_tokenize(caption), tagset="universal"):
            for tag in tags:
                if pos == tag:
                    ret.append(word)
        return ret

    def _get_phrase(self, caption, with_preposition):
        if with_preposition:
            # Taken from Su Nam Kim Paper...
            grammar = r"""
                NBAR:
                    {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

                NP:
                    {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
                    {<NBAR>} # If pattern is not found, just a single NBAR is ok
            """
        else:
            # Taken from Su Nam Kim Paper...
            grammar = r"""
                NBAR:
                    {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

                NP:
                    {<NBAR>} # If pattern is not found, just a single NBAR is ok
            """
        tokenized = self.nltk.word_tokenize(caption)
        chunker = self.nltk.RegexpParser(grammar)

        chunked = chunker.parse(self.nltk.pos_tag(tokenized))
        continuous_chunk = []
        current_chunk = []

        for subtree in chunked:
            if isinstance(subtree, self.nltk.Tree):
                current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
            elif current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue

        return continuous_chunk

    def get_noun_phrase(self, caption):
        noun_phrase = []
        noun_phrase.extend(self._get_phrase(caption, with_preposition=False))
        noun_phrase.extend(self._get_phrase(caption, with_preposition=True))

        return list(set(noun_phrase))

    def prepare_targets(self, captions, targets):

        if targets is None:
            targets = [{} for _ in range(len(captions))]

        for caption, target in zip(captions, targets):
            caption = np.random.choice(caption)
            if self.word_tags == "noun_phrase":
                words = self.get_noun_phrase(caption)
            elif "noun_phrase" in self.word_tags:
                words = []
                words.extend(self.get_noun_phrase(caption))
                words.extend(self.get_tag(caption, tuple(set(self.word_tags) - set("noun_phrase"))))
                words = list(set(words))
            else:
                words = self.get_tag(caption, self.word_tags)
            if not len(words):
                words = [""]
            # drop with probability
            words_after_drop = [w for w in words if np.random.rand() > self.word_dropout]
            if len(words_after_drop) == 0:
                # Fall back to no drop if all words are dropped
                words_after_drop = words
            words = np.random.choice(words_after_drop, size=self.num_words).tolist()
            target["words"] = words

            valid_mask = [len(w) > 0 for w in words]
            valid_mask = torch.tensor(valid_mask, device=self.device, dtype=torch.bool)
            target["word_valid_mask"] = valid_mask

        return targets

    def forward(self, outputs, targets=None):
        if self.training:
            words = [x["words"] for x in targets]

            words = prompt_labels(words, self.prompt)

            word_embed = self.build_text_embed(words)
            # [B, K, C]
            word_embed = torch.stack(word_embed.split([len(w) for w in words]), dim=0)

            word_embed = self.text_proj(word_embed)

            return {"word_embed": word_embed}
        else:
            assert targets is None
            assert self.test_labels is not None
            labels = self.test_labels

            labels = prompt_labels(labels, self.prompt)

            text_embed = self.get_and_cache_test_text_embed(labels)

            text_embed = self.text_proj(text_embed)
            return {"text_embed": text_embed, "labels": labels}


class CategoryEmbed(nn.Module):
    def __init__(
        self,
        labels,
        projection_dim,
        clip_model_name="ViT-L-14",
        prompt=None,
    ):
        super().__init__()
        self.labels = labels

        self.clip_model_name = clip_model_name
        self.clip = ClipAdapter(name=self.clip_model_name, normalize=False)

        if projection_dim < 0:
            self.text_proj = nn.Identity()
        else:
            self.text_proj = nn.Linear(self.clip.dim_latent, projection_dim)

        self.register_buffer(
            "text_embed", self.build_text_embed(prompt_labels(labels, prompt), verbose=True), False
        )
        self.null_embed = nn.Parameter(self.build_text_embed(""))

        self.prompt = prompt

        self.test_labels = None
        self._test_text_embed_dict = dict()

    def extra_repr(self) -> str:
        return f"clip_model_name={self.clip_model_name},\n"

    @property
    def device(self):
        return self.clip.device

    def _open_state_dict(self):
        return {"test_labels": self.test_labels}

    def _save_open_state_dict(self, destination, prefix):
        for k, v in self._open_state_dict().items():
            destination[prefix + k] = v

    def open_state_dict(self, destination=None, prefix=""):
        if destination is None:
            destination = OrderedDict()
        self._save_open_state_dict(destination, prefix)
        for name, module in self.named_modules(remove_duplicate=True):
            if module is self:
                continue
            if module is not None and hasattr(module, "open_state_dict"):
                module.open_state_dict(destination, prefix + name + ".")
        return destination

    @torch.no_grad()
    def build_text_embed(self, labels, verbose=False):
        return build_clip_text_embed(
            clip_model_name=self.clip.clip,
            labels=labels,
            verbose=verbose,
        )

    def get_and_cache_test_text_embed(self, labels):
        labels = to_tuple(labels)
        if labels not in self._test_text_embed_dict:
            text_embed = self.build_text_embed(labels, verbose=True)
            self._test_text_embed_dict[labels] = text_embed.cpu()
        else:
            text_embed = self._test_text_embed_dict[labels].to(self.device)
        return text_embed

    def forward(self, outputs, targets=None):
        if self.training:

            text_embed = self.text_proj(self.text_embed)
            null_embed = self.text_proj(self.null_embed)

            return {"text_embed": text_embed, "null_embed": null_embed, "labels": self.labels}

        else:
            assert targets is None
            assert self.test_labels is not None
            labels = self.test_labels
            text_embed = self.get_and_cache_test_text_embed(prompt_labels(labels, self.prompt))

            text_embed = self.text_proj(text_embed)
            null_embed = self.text_proj(self.null_embed)

            return {"text_embed": text_embed, "null_embed": null_embed, "labels": labels}



class PoolingCLIPHead(WordEmbed):
    def __init__(
        self,
        clip_model_name="ViT-L-14-336",
        alpha=0.35,
        beta=0.65,
        prompt="photo",
        train_labels=None,
        normalize_logits=True,
        bg_labels=None,
    ):
        super(WordEmbed, self).__init__()
        self.clip_model_name = clip_model_name
        # For ViT CLIP, we found MaskCLIP yields slightly better performance
        # than pooling on CLIP feature map
        self.clip = MaskCLIP(name=self.clip_model_name)

        self.alpha = alpha
        self.beta = beta

        self.test_labels = None
        self._test_text_embed_dict = dict()

        self.prompt = prompt
        if train_labels is None:
            self.train_labels = get_instance_labels("cod10k", prompt_engineered=False)
        else:
            self.train_labels = train_labels

        self.bg_labels = bg_labels
        self.normalize_logits = normalize_logits

    def extra_repr(self) -> str:
        return f"clip_model_name={self.clip_model_name},\n"

    @property
    def with_bg(self):
        return self.bg_labels is not None

    def prepare_targets(self, outputs, targets):

        target_mask_embed = self.clip.get_mask_embed(outputs["images"], outputs["pred_masks"])

        for idx in range(len(targets)):
            targets[idx]["target_mask_embed"] = target_mask_embed[idx]

        return targets

    def forward(self, outputs, targets=None):
        assert not self.training, "PoolingCLIPHead only supports inference"
        assert targets is None
        assert self.test_labels is not None
        pred_open_logits = outputs.pop("pred_open_logits")

        labels = prompt_labels(self.test_labels, self.prompt)
        if self.with_bg and pred_open_logits.shape[-1] == len(self.test_labels) + 1:
            labels.append(self.bg_labels)

        category_overlapping_list = []

        train_labels = {l for label in self.train_labels for l in label}

        for test_label in self.test_labels:
            category_overlapping_list.append(not set(train_labels).isdisjoint(set(test_label)))

        if self.with_bg and pred_open_logits.shape[-1] == len(self.test_labels) + 1:
            category_overlapping_list.append(False)

        category_overlapping_mask = torch.tensor(
            category_overlapping_list, device=outputs["images"].device, dtype=torch.long
        )

        text_embed = self.get_and_cache_test_text_embed(labels)

        mask_pred_results = outputs["pred_masks"]

        clip_results = self.clip(
            outputs["images"],
            mask_pred_results,
            text_embed,
            labels,
        )

        mask_pred_open_logits = clip_results["mask_pred_open_logits"]

        if self.normalize_logits:
            pred_open_prob = pred_open_logits.softmax(dim=-1)

            mask_pred_open_prob = mask_pred_open_logits.softmax(dim=-1)

            # NOTE: logits are multiplied with logit_scale,
            # avoid double scale, which cause overflow in softmax
            pred_open_logits_base = (
                (pred_open_prob ** (1 - self.alpha) * mask_pred_open_prob**self.alpha).log()
                # * outputs["logit_scale"]
                * category_overlapping_mask
            )

            pred_open_logits_novel = (
                (pred_open_prob ** (1 - self.beta) * mask_pred_open_prob**self.beta).log()
                # * outputs["logit_scale"]
                * (1 - category_overlapping_mask)
            )
        else:

            # NOTE: this version ignore the scale difference during ensemble,

            pred_open_logits_base = (
                pred_open_logits * (1 - self.alpha)
                + mask_pred_open_logits * self.alpha * category_overlapping_mask
            )
            pred_open_logits_novel = pred_open_logits * (
                1 - self.beta
            ) + mask_pred_open_logits * self.beta * (1 - category_overlapping_mask)

        pred_open_logits = pred_open_logits_base + pred_open_logits_novel

        ret = {"pred_open_logits": pred_open_logits}
        if "labels" in outputs:
            ret["labels"] = labels

        return ret
