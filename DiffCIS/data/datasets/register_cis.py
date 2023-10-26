import os
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog


COD10K_ROOT = '/home/anh/anh_hdd/Workspace/Camouflaged/CamoDiffusion/datasets/COD10K'
ANN_ROOT = os.path.join(COD10K_ROOT, 'annotations')
TRAIN_PATH = os.path.join(COD10K_ROOT, 'Train_Image_CAM')
TEST_PATH = os.path.join(COD10K_ROOT, 'Test_Image_CAM')
TRAIN_JSON = os.path.join(ANN_ROOT, 'train_instance.json')
TEST_JSON = os.path.join(ANN_ROOT, 'test2026.json')

NC4K_ROOT = '/home/anh/anh_hdd/Workspace/Camouflaged/CamoDiffusion/datasets/NC4K/NC4K'
NC4K_PATH = os.path.join(NC4K_ROOT, 'test/image')
NC4K_JSON = os.path.join(NC4K_ROOT, 'nc4k_test.json')

# CLASS_NAMES = ["foreground"]

PREDEFINED_SPLITS_DATASET = {
    "cod10k_train": (TRAIN_PATH, TRAIN_JSON),
    "cod10k_test": (TEST_PATH, TEST_JSON),
    "nc4k_test": (NC4K_PATH, NC4K_JSON),
}


def register_all_cod10k_instance():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key,
                                   json_file=json_file,
                                   image_root=image_root)


def register_dataset_instances(name, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type="coco")


# COD10k_CATEGORIES = {'1':'batFish','2':'clownFish','3':'crab','4':'crocodile','5':'crocodileFish','6':'fish','7':'flounder','8':'frogFish','9':'ghostPipefish','10':'leafySeaDragon','11':'octopus','12':'pagurian','13':'pipefish','14':'scorpionFish','15':'seaHorse','16':'shrimp','17':'slug','18':'starFish','19':'stingaree','20':'turtle','21':'ant','22':'bug','23':'cat','24':'caterpillar','25':'centipede','26':'chameleon','27':'cheetah','28':'deer','29':'dog','30':'duck','31':'gecko','32':'giraffe','33':'grouse','34':'human','35':'kangaroo','36':'leopard','37':'lion','38':'lizard','39':'monkey','40':'rabbit','41':'reccoon','42':'sciuridae','43':'sheep','44':'snake','45':'spider','46':'stickInsect','47':'tiger','48':'wolf','49':'worm','50':'bat','51':'bee','52':'beetle','53':'bird','54':'bittern','55':'butterfly','56':'cicada','57':'dragonfly','58':'frogmouth','59':'grasshopper','60':'heron','61':'katydid','62':'mantis','63':'mockingbird','64':'moth','65':'owl','66':'owlfly','67':'frog','68':'toad','69':'other'}


# def _get_cod10k_instances_meta():
#     thing_ids = [k for k,_ in COD10k_CATEGORIES.items()]
#     assert len(thing_ids) == 69, len(thing_ids)
#     # Mapping from the incontiguous COD10k category id to an id in [0, 69]
#     thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
#     thing_classes = [v for _,v in COD10k_CATEGORIES.items()]
#     # print(thing_ids)
#     # print(thing_classes)
#     ret = {
#         "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
#         "thing_classes": thing_classes,
#     }
#     return ret


# def register_all_cod10k_instance():
    
#     for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
#         # MetadataCatalog.remove(key)
#         # Assume pre-defined datasets live in `./datasets`.
#         register_coco_instances(
#             key,
#             _get_cod10k_instances_meta(),
#             json_file,
#             image_root,
#         )