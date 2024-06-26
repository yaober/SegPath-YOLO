model_name = "SegPath-YOLO"
model_path = "./SegPath-YOLO.onnx"
device = "cpu"
batch_size = 1
default_input_size = 640


labels = ['blood', 'dead', 'immune', 'macrophage', 'others', 'stromal', 'tumor']
labels_color = {
    -100: '#949494',
    0: '#ffffff',
    1: '#ff00ff',
    2: '#0094e1',
    3: '#0000ff',
    4: '#ffff00',
    5: '#646464',
    6: '#ff0000',
    7: '#00ff00',
}
labels_text = {
    -100: 'unlabeled',
    1: 'blood',
    2: 'dead',
    3: 'immune',
    4: 'macrophage',
    5: 'others',
    6: 'stromal',
    7: 'tumor',
}


nms_params = {
    "conf_thres": 0.25, #  0.45,  # score_threshold, discards boxes with score < score_threshold
    "iou_thres": 0.45, #  0.7,  # iou_threshold, discards all overlapping boxes with IoU > iou_threshold
    "classes": None, 
    "agnostic": True, # False
    "multi_label": False, 
    "labels": (), 
    "nc": 1,
    "max_det": 300,  # maximum detection
}


