import streamlit as st
import matplotlib.pyplot as plt
import torch
import numpy as np
import pytesseract as pt
import os
import xml.etree.ElementTree as ET
import pandas as pd
import cv2
from markdownlit import mdlit
from streamlit_extras.badges import badge
from streamlit_extras.tags import tagger_component
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header
from streamlit_toggle import st_toggle_switch
from streamlit_text_rating.st_text_rater import st_text_rater


INPUT_WIDTH =  640
INPUT_HEIGHT = 640

# load YOLOv5 model
net = cv2.dnn.readNetFromONNX('./model/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def hash_cv2(net):
    return id(net)


@st.cache_data(hash_funcs={cv2.dnn.Net:hash_cv2})
def get_detections(img, net):
    # 1.convert image to yolo model
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
    input_image[0:row,0:col] = image

    # 2.get prediction for yolo model
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]

    return input_image, detections



@st.cache_data
def non_maximum_supression(input_image, detections):

    # 3. Ffilter detection

    # center x, center y, w , h, conf, proba
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WIDTH
    y_factor = image_h/INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4] # confidence of detecting license plate
        if confidence > 0.4:
            class_score = row[5] # probability score of license plate
            if class_score > 0.01:
                cx, cy , w, h = row[0:4]

                left = int((cx - 0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left, top, width, height])

                confidences.append(confidence)
                boxes.append(box)

    # 4.1 clear
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()

    # 4.2 nms
    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)

    return boxes_np, confidences_np, index



@st.cache_data
def drawings(image, boxes_np, confidences_np, index):
    # 5. Drawings
    for ind in index:
        x, y, w, h =  boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
        accuracy = '{:.0f}'.format(bb_conf*100)
        accuracy = float(accuracy)
        license_text = extract_text(image, boxes_np[ind])
        license = license = clean_license(license_text)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.rectangle(image, (x, y-30), (x+w, y), (255, 0, 255), -1)
        cv2.rectangle(image, (x, y+h), (x+w, y+h+25), (0, 0, 0), -1)

        cv2.putText(image, conf_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(image, license,(x, y+h+27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    return accuracy, license, image



@st.cache_data
def extract_text(image, bbox):
    x,y,w,h = bbox
    roi = image[y:y+h, x:x+w]

    if 0 in roi.shape:
        return 'no number'

    else:
        text = pt.image_to_string(roi)
        text = text.strip()

        return text


@st.cache_data(hash_funcs={cv2.dnn.Net:hash_cv2})
def yolo_predictions(img, net):
    # step-1: detections
    input_image, detections = get_detections(img, net)
    # step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    # step-3: Drawings
    accuracy, license_text, result_img = drawings(img, boxes_np, confidences_np, index)

    return accuracy, license_text, result_img

@st.cache_data
def clean_license(licence):
    # Replace 'i' by 1
    licence = licence.replace('i', '1')

    # replace ';' and '.' by des spaces
    licence = licence.replace(';', '').replace('.', '').replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace('- ', '-').replace(' -', '-')

    # Remove spaces around license
    licence = licence.strip().upper()

    return licence


def main():
    st.title("IA-controller parking access authorisation")
    colored_header(
        label="Can I enter with my car ?",
        description="by Olivier Bile",
        color_name="violet-70",
    )
    badge(type="pypi", name="parking_autorization")
    # badge(type="pypi", name="tesseract")
    # badge(type="pypi", name="torch")
    mdlit("""
**Welcome to our Licence Plate Recognition Application**.

Discover the revolution in car park access management with our cutting-edge number plate recognition application, specially designed to meet your security and efficiency needs.

**How it works**

Our application is simple to use and incredibly effective:

1. **Enter Licence Plate Number**: Simply enter the licence plate number of the car you wish to check.
2. **Custom Accuracy**: Choose the level of accuracy you want for plate recognition.
3. **Decide Access**: Use our intuitive switch to decide whether access is authorised or denied.
4. **Load Photo**: Easily load the photo of the car taken from the front.

**Speed, Reliability and Security**: 

Our cutting-edge technology guarantees a quick and accurate decision, reducing waiting times and improving the security of your private car park. You can be sure that only authorised cars will gain immediate access.

*To test, upload this photo: [the 3008 SUV picture](https://imgcdn.zigwheels.ph/large/gallery/exterior/26/1826/peugeot-3008-front-medium-view-158548.jpg)*

""")

    # Add an input field
    st.write("### Edit Authorization")
    user_input = st.text_input("Enter a number plate:")

    # Add a slider
    selected_value = st.slider("Select an accuracy", 0, 100, 95)

    # Use the selected value
    st.write(f"Selected accuracy: {selected_value}%" )
    is_authorized = st_toggle_switch(
        label="This car is authorized ?",
        key="is_authorized",
        default_value=False,
        label_after=False,
        inactive_color="#D3D3D3",  # optional
        active_color="#00FF00",  # optional
        track_color="#00BB00",  # optional
    )
    # Use the value entered by the user
    uploaded_file = st.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png"])
    mdlit("""### Result:""")
    if uploaded_file is not None:
        image = plt.imread(uploaded_file)
        accuracy, license, result_img = yolo_predictions(image, net)
        st.image(result_img)
        # verify plate
        car_authorization = False
        if user_input == "":
            if license in ["GN 774-VF", "BL 600 VF"]:
                car_authorization = True
        else:
            if license == user_input and is_authorized:
                car_authorization = True
                
        if car_authorization and accuracy >= selected_value:
            tagger_component(
                "The car is ",
                ["authorized"],
                color_name=["green"],
            )
        else:
            tagger_component(
                "The car is ",
                ["not authorized"],
                color_name=["red"],
            )

        add_vertical_space(1)
        # Display plate
        st.write("the car number plate is: ", license)
        if accuracy >= selected_value:
            tagger_component(
                "the number plate reader accuracy",
                [accuracy],
                color_name=["green"],
            )
        else:
            tagger_component(
                "the number plate reader accuracy",
                [accuracy],
                color_name=["red"],
            )
        mdlit("""***""")
        # response = st_text_rater(text="If you like my project, Give me a like !")
        # st.write(f"the number plate reader accuracy: {accuracy}%")
if __name__ == "__main__":
    main()
