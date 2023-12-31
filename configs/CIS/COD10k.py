
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler
from ..common.data.cod10k_instance_seg import dataloader
from ..common.models.diffcis_with_label import model

from ..common.train import train
from ..common.optim import AdamW as optimizer

from ..common.data.nc4k_eval import (
    nc4k_eval as _nc4k_eval
)


train.max_iter = 92_188
train.grad_clip = 0.01
train.checkpointer.period = 50

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        # assume 100e with batch-size 64 as original LSJ
        # Equivalent to 100 epochs.
        # 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
        milestones=[163889, 177546],
        num_updates=184375,
    ),
    # for warmup length we adopted COCO LSJ setting
    warmup_length= train.checkpointer.period / 184375,
    warmup_factor=0.067,
)

optimizer.lr = 1e-4
optimizer.weight_decay = 0.05

# dataloader.extra_task = dict(
#     eval_nc4k=_nc4k_eval
# )

def TVA_module(image_features, text_features):

    # weights to restrain influence of obvious classes on others
    prob = image_features[:, :1, :] @ text_features.t()
    prob = (prob * 2).softmax(-1)
    w = prob / prob.mean(-1, keepdim=True)

    # element-wise multiplied features
    b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], image_features.shape[1], image_features.shape[2]
    feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(1, 1, n_t, c)
    feats *= w.reshape(1, 1, n_t, 1)
    redundant_feats = feats.mean(2, keepdim=True) # along cls dim
    feats = feats - redundant_feats
    
    # sum the element-wise multiplied features as cosine similarity
    similarity = feats.sum(-1)

    return similarity