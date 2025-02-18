# import torch
# import gradio as gr
# from read import text_recognizer
# from model import Model
# from utils import CTCLabelConverter
# from ultralytics import YOLO
# from PIL import ImageDraw

# """ vocab / character number configuration """
# file = open("UrduGlyphs.txt","r",encoding="utf-8")
# content = file.readlines()
# content = ''.join([str(elem).strip('\n') for elem in content])
# content = content+" "
# """ model configuration """
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# converter = CTCLabelConverter(content)
# recognition_model = Model(num_class=len(converter.character), device=device)
# recognition_model = recognition_model.to(device)
# recognition_model.load_state_dict(torch.load("best_norm_ED.pth", map_location=device))
# recognition_model.eval()

# detection_model = YOLO("yolov8m_UrduDoc.pt")

# examples = ["1.jpg","2.jpg","3.jpg"]

# input = gr.Image(type="pil",image_mode="RGB", label="Input Image")

# def predict(input):
#     "Line Detection"
#     detection_results = detection_model.predict(source=input, conf=0.2, imgsz=1280, save=False, nms=True, device=device)
#     bounding_boxes = detection_results[0].boxes.xyxy.cpu().numpy().tolist()
#     bounding_boxes.sort(key=lambda x: x[1])
    
#     "Draw the bounding boxes"
#     draw = ImageDraw.Draw(input)
#     for box in bounding_boxes:
#         # draw rectangle outline with random color and width=5
#         from numpy import random
#         draw.rectangle(box, fill=None, outline=tuple(random.randint(0,255,3)), width=5)
    
#     "Crop the detected lines"
#     cropped_images = []
#     for box in bounding_boxes:
#         cropped_images.append(input.crop(box))
#     len(cropped_images)
    
#     "Recognize the text"
#     texts = []
#     for img in cropped_images:
#         texts.append(text_recognizer(img, recognition_model, converter, device))
    
#     "Join the text"
#     text = "\n".join(texts)
    
#     "Return the image with bounding boxes and the text"
#     return input,text

# output_image = gr.Image(type="pil",image_mode="RGB",label="Detected Lines")
# output_text = gr.Textbox(label="Recognized Text",interactive=True,show_copy_button=True)

# iface = gr.Interface(predict,
#                      inputs=input,
#                      outputs=[output_image,output_text],
#                      title="End-to-End Urdu OCR",
#                      description="Demo Web App For UTRNet\n(https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition)",
#                      examples=examples,
#                      allow_flagging="never")
# iface.launch()



import torch
from pathlib import Path
import gradio as gr
from read import text_recognizer
from model import Model
from utils import CTCLabelConverter
from ultralytics import YOLO
from PIL import ImageDraw
import PIL
from typing import List, Tuple
import numpy as np

def load_models(recognition_model_path: str, detection_model_path: str) -> Tuple[Model, YOLO]:
    """
    Safely load the recognition and detection models with proper error handling
    """
    try:
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load vocabulary
        with open("UrduGlyphs.txt", "r", encoding="utf-8") as file:
            content = ''.join(line.strip() for line in file) + " "
        
        # Initialize recognition model
        converter = CTCLabelConverter(content)
        recognition_model = Model(num_class=len(converter.character), device=device)
        recognition_model = recognition_model.to(device)
        
        # Safely load recognition model weights
        if Path(recognition_model_path).exists():
            recognition_model.load_state_dict(
                torch.load(recognition_model_path, map_location=device, weights_only=True)
            )
        else:
            raise FileNotFoundError(f"Recognition model not found at {recognition_model_path}")
        
        recognition_model.eval()
        
        # Load detection model
        if Path(detection_model_path).exists():
            detection_model = YOLO(detection_model_path)
        else:
            raise FileNotFoundError(f"Detection model not found at {detection_model_path}")
        
        return recognition_model, detection_model, converter, device
    
    except Exception as e:
        raise RuntimeError(f"Error loading models: {str(e)}")

def predict(input_image: PIL.Image.Image) -> Tuple[PIL.Image.Image, str]:
    """
    Process image through detection and recognition pipeline
    """
    try:
        # Line Detection
        detection_results = detection_model.predict(
            source=input_image, 
            conf=0.2, 
            imgsz=1280, 
            save=False, 
            nms=True, 
            device=device
        )
        
        # Sort bounding boxes by y-coordinate
        bounding_boxes = detection_results[0].boxes.xyxy.cpu().numpy().tolist()
        bounding_boxes.sort(key=lambda x: x[1])
        
        # Draw bounding boxes
        draw = ImageDraw.Draw(input_image)
        for box in bounding_boxes:
            color = tuple(np.random.randint(0, 255, 3).tolist())
            draw.rectangle(box, fill=None, outline=color, width=5)
        
        # Process each detected line
        texts = []
        for box in bounding_boxes:
            cropped_image = input_image.crop(box)
            text = text_recognizer(cropped_image, recognition_model, converter, device)
            texts.append(text)
        
        return input_image, "\n".join(texts)
    
    except Exception as e:
        raise RuntimeError(f"Error processing image: {str(e)}")

# Initialize models
recognition_model, detection_model, converter, device = load_models(
    "best_norm_ED.pth",
    "yolov8m_UrduDoc.pt"
)

# Setup Gradio interface
examples = ["1.jpg", "2.jpg", "3.jpg"]

with gr.Blocks(title="End-to-End Urdu OCR") as iface:
    gr.Markdown("""
    # End-to-End Urdu OCR
    Demo Web App For UTRNet
    (https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition)
    """)
    
    with gr.Row():
        input_image = gr.Image(
            type="pil",
            image_mode="RGB",
            label="Input Image"
        )
        output_image = gr.Image(
            type="pil",
            image_mode="RGB",
            label="Detected Lines"
        )
    
    output_text = gr.Textbox(
        label="Recognized Text",
        interactive=True,
        show_copy_button=True
    )
    
    gr.Examples(
        examples=examples,
        inputs=input_image,
        outputs=[output_image, output_text],
        fn=predict,
        cache_examples=True
    )
    
    input_image.change(
        fn=predict,
        inputs=input_image,
        outputs=[output_image, output_text]
    )

if __name__ == "__main__":
    iface.launch()