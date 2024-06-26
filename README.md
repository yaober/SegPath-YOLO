# SegPath-YOLO
A High-Speed, High-Accuracy Pathology Image Segmentation and Tumor Microenvironment Feature Extraction Tool

## Description
SegPath-YOLO addresses critical challenges in pathology image analysis, particularly in handling overlapping and high-density cellular structures and ensuring rapid processing without sacrificing precision. The novelty of SegPath-YOLO lies in its Segmentation and Overlapping-Aware Loss, which utilizes a binary overlap mask to identify and enhance the loss in overlapping regions. In conjunction with PathNuclei attention mechanisms, SegPath-YOLO not only refines segmentation results but also contributes to a deeper characterization and quantification of the tumor microenvironment, significantly aiding in survival outcome predictions. 
## Getting Started

### Dependencies

* Python 3.8+
* PyTorch 1.8.0+
* Gradio 4.20.+
* Other Python libraries as specified in requirements.txt
### Installing

* Downlaod SegPath-YOLO from Gihtub
``` bash
git clone https://github.com/yaober/SegPath-YOLO.git
```
* Pip install the ultralytics package
``` bash
pip install -r requirements.txt
```

### Executing program

* The source code of SegPath-YOLO will be released when the paper is accepted.

## Demo

Run Gradio for the interactive demo:

``` bash
python app_gradio.py
```

## Authors

Contributors names and contact info

* Mr. Jia Yao  
* Dr. Ruichen Rong
* Dr. Tao Wang
* Dr. Guanghua Xiao

