from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import get_detection_dataset_dicts
from detectron2.data import MetadataCatalog

from DiffCIS.modeling.wrapper.instance_wrapper import CISInference
from DiffCIS.data.build import get_instance_labels
from DiffCIS.data import (
    DatasetMapper_DiffCIS,
    build_d2_test_dataloader
)

from DiffCIS.evaluation.d2_evaluator import (
    InstanceSegEvaluator
)


nc4k_eval = OmegaConf.create()
nc4k_eval.loader = L(build_d2_test_dataloader)(
    dataset=L(get_detection_dataset_dicts)(names="nc4k_test", filter_empty=False),
    mapper=L(DatasetMapper_DiffCIS)(
        is_train=False,
        tfm_gens=[
            T.ResizeScale(
            min_scale=1, max_scale=1, target_height=1024, target_width=1024
        )],
        image_format="RGB",
    ),
    local_batch_size=1,
    num_workers=1,
)

nc4k_eval.wrapper = L(CISInference)(
    labels=L(get_instance_labels)(dataset="nc4k", prompt_engineered=False),
    metadata=L(MetadataCatalog.get)(name=nc4k_eval.loader.dataset.names),
)

nc4k_eval.evaluator = [
    L(InstanceSegEvaluator)(
        dataset_name="${...loader.dataset.names}",
        tasks=("segm",),
    )
]

