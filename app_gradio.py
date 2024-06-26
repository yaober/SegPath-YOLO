import time
import numpy as np
import gradio as gr
import pandas as pd

from PIL import Image
from io import BytesIO
from models.utils import export_detections_to_image, export_detections_to_table

import configs
from models import Yolov8SegmentationONNX

service = Yolov8SegmentationONNX(configs, device='cpu')


def predict(images):
    images = [np.array(img) for img in images]
    st = time.time()
    results = service(images, preprocess=True)
    print(f"Inference batch (size={len(images)}): {time.time() - st}s.")

    masks, dataframes = [], []
    for res, img in zip(results, images):
        ## save tables
        st = time.time()
        df = export_detections_to_table(
            res, labels_text=service.configs.labels_text,
            save_masks=True,
        )
        print(f"Export table: {time.time() - st}s.")
        dataframes.append(df)

        ## save image and overlay masks
        st = time.time()
        mask_img = export_detections_to_image(
            res, (img.shape[0], img.shape[1]), 
            labels_color=service.configs.labels_color,
            save_masks=True, border=3, alpha=0.3,
        )
        print(f"Export image: {time.time() - st}s.")
        masks.append(mask_img)

#         export_img = overlay_masks_on_image(img, mask_img)
#         buf = BytesIO()
#         export_img.save(buf, format='png', quality=100)
#         export_images.append(buf.getvalue())

    return [masks, dataframes]


def visualize(image, mask):
    blended = image.copy()
    blended.paste(mask, mask=mask.split()[-1])

    return blended


with gr.Blocks() as demo:
    with gr.Row(equal_height=True):
        inputs = gr.Image(label='inputs', image_mode='RGB', type='pil')
        outputs = gr.Image(label='outputs', image_mode='RGBA', interactive=False, type='pil')  # shape=(320, 320)
    with gr.Row(equal_height=True):
        with gr.Column():
            with gr.Row(equal_height=True):
                submit_btn = gr.Button("Run")
                clear_btn = gr.Button("Clear")
            display_image = gr.Image(label='overlay', image_mode='RGBA', interactive=False, type='pil')
        with gr.Column():
            display_table = gr.Dataframe(
                label='detections',
                headers=['x0', 'y0', 'x1', 'y1', 'label', 'score', 'poly_x', 'poly_y'],
            )
    submit_btn.click(
        fn=predict, inputs=[inputs], outputs=[outputs, display_table], 
        api_name='predict', batch=True, max_batch_size=16).success(  # concurrency_limit=1, 
        fn=visualize, inputs=[inputs, outputs], outputs=[display_image])
    clear_btn.click(lambda: [None]*4, outputs=[inputs, outputs, display_image, display_table])

demo.queue(api_open=True)
demo.launch(share=True, server_name='0.0.0.0', app_kwargs={"docs_url": "/docs"})  # server_port=7861, 
