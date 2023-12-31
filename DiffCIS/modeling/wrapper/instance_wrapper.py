
from collections import OrderedDict
import torch.nn as nn


class CISInference(nn.Module):
    def __init__(
        self,
        model,
        labels,
        metadata=None,
        semantic_on=False,
        instance_on=True,
        panoptic_on=False,
        test_topk_per_image=10,
    ):
        super().__init__()
        self.model = model
        self.labels = labels
        self.metadata = metadata

        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        self.open_state_dict = OrderedDict()

        for k in self.model.open_state_dict():
            if k.endswith("test_labels"):
                self.open_state_dict[k] = self.labels
            elif k.endswith("metadata"):
                self.open_state_dict[k] = self.metadata
            elif k.endswith("num_classes"):
                self.open_state_dict[k] = self.num_classes
            elif k.endswith("semantic_on"):
                self.open_state_dict[k] = self.semantic_on
            elif k.endswith("instance_on"):
                self.open_state_dict[k] = self.instance_on
            elif k.endswith("panoptic_on"):
                self.open_state_dict[k] = self.panoptic_on
            elif k.endswith("test_topk_per_image"):
                self.open_state_dict[k] = self.test_topk_per_image

    @property
    def num_classes(self):
        return len(self.labels)

    def forward(self, batched_inputs):
        assert not self.training

        _open_state_dict = self.model.open_state_dict()
        self.model.load_open_state_dict(self.open_state_dict)

        results = self.model(batched_inputs)

        self.model.load_open_state_dict(_open_state_dict)

        return results
