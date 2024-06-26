# """ YOlov8 model configuration"""
# import os
# from collections import OrderedDict
# from typing import Mapping

# # from ...configuration_utils import ModelConfig
# # from ...onnx import OnnxConfig
# # from ...utils import logging
# # logger = logging.get_logger(__name__)


# YOLOv8_PRETRAINED_CONFIG_ARCHIVE_MAP = {
#     "yolov8-lung-nuclei": "../../configs/yolov8-lung/config.json",
# }


# class Yolov8SegmentationConfig:
#     model_type: str = ""
#     attribute_map: Dict[str, str] = {}

#     def __setattr__(self, key, value):
#         self.attribute_map[key] = value

#     def __getattribute__(self, key):
#         return self.attribute_map[key]

#     def __init__(self, **kwargs):
#         for k, v in kwargs.items():
#             self.attribute_map[k] = v

#     @property
#     def num_labels(self) -> int:
#         return len(self.labels)

#     @classmethod
#     def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> ModelConfig:
#         config_dict = {**config_dict, **kwargs}
#         return cls(**config_dict)

#     def to_dict(self) -> Dict[str, Any]:
#         return self.attribute_map

#     @classmethod
#     def from_json_file(cls, json_file: Union[str, os.PathLike]) -> ModelConfig:
#         with open(json_file, "r", encoding="utf-8") as reader:
#             text = reader.read()
#         config_dict = json.loads(text)
#         return cls(**config_dict)

#     def to_json_string(self) -> str:
#         config_dict = self.to_dict()
#         return json.dumps(config_dict, indent=4, sort_keys=True) + "\n"

#     def to_json_file(self, json_file_path: Union[str, os.PathLike]):
#         with open(json_file_path, "w", encoding="utf-8") as f:
#             f.write(self.to_json_string())

#     def update(self, config_dict: Dict[str, Any]):
#         for key, value in config_dict.items():
#             setattr(self, key, value)
    
#     def model_config(self):
#         pass
    
#     def service_config(self):
#         pass
    
#     def dataset_config(self):
#         pass
        


# DEFAULT_MPP = 0.25
# PATCH_SIZE = 512
# PADDING = 64
# PAGE = 0
# MASK_ALPHA = 0.3

# MODEL_PATHS = {
#     'lung': './ckpts/yolov8/lung.best.torchscript',
# }

# DATASETS = {
#     'patch_size': PATCH_SIZE,
#     'padding': PADDING,
#     'page': PAGE,
#     'labels': ['bg', 'tumor', 'stromal', 'immune', 'blood', 'macrophage', 'dead', 'other',],
#     'labels_color': {
#         -100: '#949494',
#         0: '#ffffff', 
#         1: '#00ff00', 
#         2: '#ff0000', 
#         3: '#0000ff', 
#         4: '#ff00ff', 
#         5: '#ffff00',
#         6: '#0094e1',
#         7: '#646464',
#     },
#     'labels_text': {
#         0: 'bg', 1: 'tumor', 2: 'stromal', 3: 'immune', 
#         4: 'blood', 5: 'macrophage', 6: 'dead', 7: 'other',
#     },
# }

# NMS_PARAMS = {
#     'conf_thres': 0.15, # 0.25, # 0.15,  # score_threshold, discards boxes with score < score_threshold
#     'iou_thres': 0.45, # 0.7, # 0.45,  # iou_threshold, discards all overlapping boxes with IoU > iou_threshold
#     'classes': None, 
#     'agnostic': True, # False
#     'multi_label': False, 
#     'labels': (), 
#     'nc': 7,
#     'max_det': 500,  # maximum detection
# }

# ROI_NAMES = {
#     'tissue': True,  # use tissue region as roi
#     'xml': '.*',  # use all annotations in slide_id.xml 
# }

# TIFF_PARAMS = {
#     'tile': (1, 256, 256), 
#     'photometric': 'RGB',
#     'compress': True,
#     'compression': 'zlib', # compression=('jpeg', 95),  # None RGBA, requires imagecodecs
# }