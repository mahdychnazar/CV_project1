
#streamlit run main.py

import streamlit as st
import torch
import pandas as pd
import numpy as np
import os, urllib, cv2


def main():
    run_app()
    return

def run_app():
    #s@st.cache_data
    def run():
        image = load_image('F:/CV/export/images/test/1478019975180844551_jpg.rf.067fd6f8346d62a38da50548d06554f6.jpg')
        boxes = load_boxes('F:/CV/export/labels/test/1478019975180844551_jpg.rf.067fd6f8346d62a38da50548d06554f6.txt')

        model = load_model("F:/CV/CV_project/model/yolov5s.pt", "cpu")

        confidence, overlap = detector_props_ui()
        boxes_pred = predict_image(image, model, confidence, overlap, 512)


        #print(image)
        #print(boxes)
        #st.image(image.astype(np.uint8), use_column_width=True)
        draw_image_boxes(image, boxes, "","")
        draw_image_boxes(image, boxes_pred, "", "")

        return
    run()
    return



def load_boxes(path):
    yolo_boxes = pd.read_csv(path, sep =" ", header=None)
    #print(yolo_boxes)
    pascal_boxes = pd.DataFrame({'label':[], 'xmin':[], 'ymin':[], 'xmax':[], 'ymax':[]})
    for i in yolo_boxes.index:
        pascal_boxes.loc[i] = yolo_to_pascal_voc(yolo_boxes[0][i],
                                                 512,
                                                 512,
                                                 yolo_boxes[1][i],
                                                 yolo_boxes[2][i],
                                                 yolo_boxes[3][i],
                                                 yolo_boxes[4][i])
    #print(pascal_boxes)
    return pascal_boxes


def load_image(path):
    with open(path, 'rb') as im:
        f = im.read()
        image = np.asarray(bytearray(f), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = image[:, :, [2, 1, 0]] # BGR -> RGB
        return image

@st.cache_resource
def load_model(path, device):
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    model_.to(device)
    print("model to ", device)
    return model_


def predict_image(img, model, confidence, iou, size=None):
    model.conf = confidence
    model.iou = iou
    #print(model)
    result = model(img, size=size) if size else model(img)
    #print(result.xyxy[0].numpy())
    prediction = result.pandas().xyxy[0]
    cols = ['class', 'xmin', 'ymin', 'xmax', 'ymax']
    confs = prediction['confidence']
    boxes = prediction[cols]
    boxes.rename(columns = {'class':'label'}, inplace = True)
    #print(boxes)
    return boxes

def draw_image_boxes(image, boxes, header, description):
    # Superpose the semi-transparent object detection boxes.    # Colors for the boxes

    LABEL_COLORS = {
        0: [255, 0, 255],
        1: [255, 0, 0],
        2: [0, 255, 0],
        3: [255, 255, 0],
        4: [255, 255, 0],
        5: [255, 255, 0],
        6: [255, 255, 0],
        7: [255, 255, 0],
        8: [255, 255, 0],
        9: [255, 255, 0],
        10: [0, 0, 255],
    }

    image_with_boxes = image.astype(np.float64)
    for _, (label, xmin, ymin, xmax, ymax) in boxes.iterrows():
        image_with_boxes[int(ymin):int(ymax), int(xmin):int(xmax), :] += LABEL_COLORS[label]
        image_with_boxes[int(ymin):int(ymax), int(xmin):int(xmax), :] /= 2

    # Draw the header and image.
    #st.subheader(header)
    #st.markdown(description)
    st.image(image_with_boxes.astype(np.uint8), use_column_width=True)


def detector_props_ui():
    st.sidebar.markdown("# Model")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold



def yolo_to_pascal_voc(class_id, width, height, x, y, w, h):

    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)
    label = int(class_id)
    return {'label' :label, 'xmin':xmin, 'ymin':ymin, 'xmax':xmax, 'ymax':ymax}

if __name__ == "__main__":
    main()

