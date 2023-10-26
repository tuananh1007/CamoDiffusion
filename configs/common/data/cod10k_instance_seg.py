
from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import get_detection_dataset_dicts
from detectron2.data import DatasetMapper
from detectron2.data import MetadataCatalog

from DiffCIS.modeling.wrapper.instance_wrapper import CISInference
from DiffCIS.data.build import get_instance_labels
from DiffCIS.data import (
    DatasetMapper_DiffCIS,
    build_d2_train_dataloader,
    build_d2_test_dataloader
)

from DiffCIS.evaluation.d2_evaluator import (
    InstanceSegEvaluator
)


dataloader = OmegaConf.create()

dataloader.train = L(build_d2_train_dataloader)(
    dataset=L(get_detection_dataset_dicts)(
        names="cod10k_train", filter_empty=True
    ),
    mapper=L(DatasetMapper_DiffCIS)(
        is_train=True,
        tfm_gens=[
        T.ResizeScale(
            min_scale=0.1, max_scale=2.0, target_height=512, target_width=512
        ),
        T.FixedSizeCrop(crop_size=(512, 512)),
    ],
        image_format="RGB",
    ),
    total_batch_size=64,
    num_workers=8,
)

dataloader.test = L(build_d2_test_dataloader)(
    dataset=L(get_detection_dataset_dicts)(
        names="cod10k_test",
        filter_empty=False,
    ),
    mapper=L(DatasetMapper_DiffCIS)(
        is_train=False,
        tfm_gens=[
            T.ResizeScale(
            min_scale=1, max_scale=1, target_height=512, target_width=512
        )],
        image_format="${...train.mapper.image_format}",
    ),
    local_batch_size=1,
    num_workers=1,
)

dataloader.wrapper = L(CISInference)(
    labels=L(get_instance_labels)(dataset="cod10k", prompt_engineered=False),
    metadata=L(MetadataCatalog.get)(name="${...test.dataset.names}"),
)

dataloader.evaluator = [
    L(InstanceSegEvaluator)(
        dataset_name="${...test.dataset.names}",
        tasks=("segm",),
    )
]

