import torch
from PIL import Image
from .model import Model
from .utils import CTCLabelConverter
from ultralytics import YOLO
from .read import text_recognizer
import fitz
import yaml
import os

pwd = os.getcwd()

def initialize_models():
    # Device configuration
    # device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    
    # Load vocabulary
    with open(os.path.join(pwd,"UrduGlyphs.txt"), "r", encoding="utf-8") as file:
        content = ''.join(line.strip() for line in file) + " "
    
    # Initialize recognition model
    converter = CTCLabelConverter(content)
    recognition_model = Model(num_class=len(converter.character), device=device)
    recognition_model = recognition_model.to(device)
    
    # Load model weights
    recognition_model.load_state_dict(
        torch.load("best_norm_ED.pth", map_location=device)
    )
    recognition_model.eval()
    
    # Load detection model
    detection_model = YOLO("yolov8m_UrduDoc.pt")
    
    return recognition_model, detection_model, converter, device


recognition_model, detection_model, converter, device = initialize_models()
print(device)

def recognize_text(image:Image):
    # Initialize models
    
    # Load image
    image = image
    
    # Detect lines
    detection_results = detection_model.predict(
        source=image, 
        conf=0.2, 
        imgsz=1280, 
        save=False, 
        nms=True, 
        device=device
    )
    
    # Sort bounding boxes by y-coordinate
    bounding_boxes = detection_results[0].boxes.xyxy.cpu().numpy().tolist()
    bounding_boxes.sort(key=lambda x: x[1])
    
    # Process each detected line
    texts = []
    for box in bounding_boxes:
        cropped_image = image.crop(box)
        text = text_recognizer(cropped_image, recognition_model, converter, device)
        texts.append(text)
    
    return "\n".join(texts)

if __name__ == "__main__":
    # Example usage
    image_path = "3.jpg"
    image = Image.open(image_path).convert('RGB')
    output_text = recognize_text(image = image)
    with open('output.txt',mode='w') as file:
        file.write(output_text)
    print(output_text)