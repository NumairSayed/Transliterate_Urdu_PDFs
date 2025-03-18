from .inference import recognize_text
import fitz
from tqdm.auto import tqdm
import fitz
from PIL import Image
from pathlib import Path
import os

# Get the path of the current file
current_file = Path(__file__)

# Get the parent directory
parent_directory = current_file.parent

def pdf_to_pil_images(pdf_path):
    """Converts a PDF file to a list of PIL Image objects."""

    pdf_document = fitz.open(pdf_path)
    images = []

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    pdf_document.close()
    return images

def main():
    # Example usage
    pdf_file = os.path.join(parent_directory,"anware_shariah.pdf")
    image_list = pdf_to_pil_images(pdf_file)
    page_wise_recognition = []
    for image in tqdm(image_list):
        text = recognize_text(image=image)
        page_wise_recognition.extend(text)
    with open('anware_shariat.txt',mode='a') as file:
        for page_no,text in enumerate(page_wise_recognition):
            file.writelines('-'*10+f' Page {page_no} '+'-'*10)
            file.writelines(text)


if __name__ == '__main__':
    main()        
    


# Now you can access each page as a PIL Image object
# for img in image_list[5:7:]:
#     img.show()  # Display the image