
from detectron2.config import LazyCall as L
from DiffCIS.modeling.meta_arch.ldm import LdmImplicitCaptionerExtractor
from DiffCIS.modeling.backbone.feature_extractor import FeatureExtractorBackbone
from .mask_generator_with_label import model

model.backbone = L(FeatureExtractorBackbone)(
    feature_extractor=L(LdmImplicitCaptionerExtractor)(
        encoder_block_indices=(5, 7),
        unet_block_indices=(2, 5, 8, 11),
        decoder_block_indices=(2, 5),
        steps=(0,),
        learnable_time_embed=True,
        num_timesteps=1,
        # clip_model_name="ViT-L-14-336",
        clip_model_name="ViT-B-16",
    ),
    out_features=["s2", "s3", "s4", "s5"],
    # out_features=["res2", "res3", "res4", "res5"],
    use_checkpoint=True,
    slide_training=True,
)
model.sem_seg_head.pixel_decoder.transformer_in_features = ["s3", "s4", "s5"]
# model.sem_seg_head.pixel_decoder.transformer_in_features = ["res3", "res4", "res5"]
model.clip_head.alpha = 0.3
model.clip_head.beta = 0.7
