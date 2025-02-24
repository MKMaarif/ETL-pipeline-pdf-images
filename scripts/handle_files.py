from pdf2image import convert_from_path
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from dotenv import load_dotenv
import os

load_dotenv()

# load the YOLO11 model
model = YOLO(os.getenv("YOLO_MODEL_PATH"))

# Function to save uploaded file
def save_uploaded_file(uploaded_file, save_path):
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

# Function to split PDF into images
def split_pdf(pdf_path):
    pages = convert_from_path(pdf_path, 200)
    page_files = []
    for i, page in enumerate(pages):
        page_path = f"upload/img/pages/page_{i+1}.png"
        page.save(page_path, "PNG")
        page_files.append(page_path)
    return page_files

# detect text, table, and figure using YOLOv11 model
def detect_text(pages, model=model):
    annotated_images = []
    texts = []
    tables = []
    figures = []
    for j, page in enumerate(pages):
        image = cv2.imread(page)
        results = model(image, conf=0.35, iou=0.7)[0]
        detections = sv.Detections.from_ultralytics(results)

        # save annotated image
        annotated_image = image.copy()
        annotated_image = sv.BoxAnnotator().annotate(scene=annotated_image, detections=detections)
        annotated_image = sv.LabelAnnotator().annotate(scene=annotated_image, detections=detections)
        output_image_path = f"upload/img/annotated/page_{j+1}_annotated.png"
        cv2.imwrite(output_image_path, annotated_image)
        annotated_images.append(output_image_path)

        # make a white image with the same size as the original image
        image_copy = np.ones_like(image) * 255

        # Iterate through detections and process sections
        for i, class_name in enumerate(detections.data['class_name']):
            if class_name == 'text':
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, detections.xyxy[i])

                # Extract the section
                section = image[y1:y2, x1:x2]

                # copy the text section to the white image
                image_copy[y1:y2, x1:x2] = section

            elif class_name in ['table', 'figure']:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, detections.xyxy[i])

                # Extract the section
                section = image[y1:y2, x1:x2]

                # Save the extracted section
                output_filename = f"upload/img/{class_name}s/page_{j+1}_{class_name}_{i}.png"
                cv2.imwrite(output_filename, section)

                # Append the section to the list of tables or figures
                if class_name == 'table':
                    tables.append(output_filename)
                elif class_name == 'figure':
                    figures.append(output_filename)
                
        # Save the modified image
        output_image_path = f"upload/img/texts/page_{j+1}_text.png"
        cv2.imwrite(output_image_path, image_copy)
        texts.append(output_image_path)

    return annotated_images, texts, tables, figures