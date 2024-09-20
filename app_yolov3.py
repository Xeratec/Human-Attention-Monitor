from pynq_dpu import DpuOverlay
overlay = DpuOverlay("./dpu.bit")

import cv2
import base64
import asyncio
import websockets
from PIL import Image
import json
import numpy as np

# Load YOLOv3 model for object detection
overlay.load_model("tf_yolov3_voc.xmodel")

# Get DPU runner
dpu = overlay.runner

# Get input/output tensor shapes
inputTensors = dpu.get_input_tensors()
outputTensors = dpu.get_output_tensors()

shapeIn = tuple(inputTensors[0].dims)
shapeOut0 = tuple(outputTensors[0].dims)  # (1, 13, 13, 75)
shapeOut1 = tuple(outputTensors[1].dims)  # (1, 26, 26, 75)
shapeOut2 = tuple(outputTensors[2].dims)  # (1, 52, 52, 75)

# Initialize input and output data
input_data = [np.empty(shapeIn, dtype=np.float32, order="C")]
output_data = [np.empty(shapeOut0, dtype=np.float32, order="C"),
               np.empty(shapeOut1, dtype=np.float32, order="C"),
               np.empty(shapeOut2, dtype=np.float32, order="C")]

anchor_list = [10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326]
anchor_float = [float(x) for x in anchor_list]
anchors = np.array(anchor_float).reshape(-1, 2)

# Preprocessing function
def preprocess_image(img, target_size):
    ih, iw = target_size
    h, w, _ = img.shape
    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(img, (nw, nh))
    image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    image_padded[(ih - nh) // 2: (ih - nh) // 2 + nh, (iw - nw) // 2: (iw - nw) // 2 + nw, :] = image_resized
    return image_padded / 255.0

def letterbox_image(image, size):
    ih, iw, _ = image.shape
    w, h = size
    scale = min(w/iw, h/ih)
    #print(scale)
    
    nw = int(iw*scale)
    nh = int(ih*scale)
    #print(nw)
    #print(nh)

    image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_LINEAR)
    new_image = np.ones((h,w,3), np.uint8) * 128
    h_start = (h-nh)//2
    w_start = (w-nw)//2
    new_image[h_start:h_start+nh, w_start:w_start+nw, :] = image
    return new_image

def _get_feats(feats, anchors, num_classes, input_shape):
    num_anchors = len(anchors)
    anchors_tensor = np.reshape(np.array(anchors, dtype=np.float32), [1, 1, 1, num_anchors, 2])
    grid_size = np.shape(feats)[1:3]
    nu = num_classes + 5
    predictions = np.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, nu])
    grid_y = np.tile(np.reshape(np.arange(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
    grid_x = np.tile(np.reshape(np.arange(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
    grid = np.concatenate([grid_x, grid_y], axis = -1)
    grid = np.array(grid, dtype=np.float32)
    
    box_xy = (1/(1+np.exp(-predictions[..., :2])) + grid) / np.array(grid_size[::-1], dtype=np.float32)
    box_wh = np.exp(predictions[..., 2:4]) * anchors_tensor / np.array(input_shape[::-1], dtype=np.float32)
    box_confidence = 1/(1+np.exp(-predictions[..., 4:5]))
    box_class_probs = 1/(1+np.exp(-predictions[..., 5:]))
    return box_xy, box_wh, box_confidence, box_class_probs


def correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape, dtype=np.float32)
    image_shape = np.array(image_shape, dtype=np.float32)
    new_shape = np.around(image_shape * np.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale
    
    print(f"Input Shape: {input_shape}")
    print(f"Image Shape: {image_shape}")
    print(f"New Shape: {new_shape}")
    print(f"Offset: {offset}")
    print(f"Scale: {scale}")

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([
        box_mins[..., 0:1],
        box_mins[..., 1:2],
        box_maxes[..., 0:1],
        box_maxes[..., 1:2]
    ], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2-x1+1)*(y2-y1+1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 1)
        h1 = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= 0.55)[0]  # threshold
        order = order[inds + 1]

    return keep

anchor_list = [10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326]
anchor_float = [float(x) for x in anchor_list]
anchors = np.array(anchor_float).reshape(-1, 2)

def postprocess_boxes(yolo_outputs, original_image_shape, input_shape, score_threshold, anchors=anchors, num_classes=20):
    boxes, scores, classes = [], [], []
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    # Process each output grid (feats) separately
    for i in range(len(yolo_outputs)):
        # Use the helper functions to extract box coordinates, scores, and class probabilities
        box_xy, box_wh, box_confidence, box_class_probs = _get_feats(yolo_outputs[i], anchors[anchor_mask[i]], num_classes, input_shape)
        
        
        # Convert box format and adjust for image scaling
        boxes_scaled = correct_boxes(box_xy, box_wh, input_shape, original_image_shape)

        # Compute box scores and filter by confidence threshold
        box_scores = box_confidence * box_class_probs
        box_classes = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)

        # Filter boxes based on the confidence score and keep 'person' class (assumed to be class 0)
        mask = box_class_scores >= score_threshold
        filtered_boxes = boxes_scaled[mask]
        filtered_scores = box_class_scores[mask]
        filtered_classes = box_classes[mask]
        
        if len(filtered_boxes) > 0:
            nms_indices = nms_boxes(filtered_boxes, filtered_scores)
            filtered_boxes = filtered_boxes[nms_indices]
            filtered_scores = filtered_scores[nms_indices]
            filtered_classes = filtered_classes[nms_indices]
        
        # Only keep boxes for class 0 (person)
        for i in range(len(filtered_boxes)):
            if filtered_classes[i] == 14:  # Assuming class 0 is 'person'
                boxes.append(filtered_boxes[i])
                scores.append(filtered_scores[i])
                classes.append(filtered_classes[i])

    return boxes, scores, classes



# Function to encode the image in Base64 from an image variable
def process_image(img):
    if img is None or img.size == 0:
        print("Empty image received.")
        return None
    
    result, buffered = cv2.imencode('.jpg', img)
    if result:
        img_base64 = base64.b64encode(buffered).decode('utf-8')
        return img_base64
    else:
        print("Error during image encoding")
        return None

# WebSocket server handler to send the image
async def send_image(websocket, path):
    cam = cv2.VideoCapture(0)
    print("Client connected")
    
    while True:
        # Capture frame from the webcam
        result, image = cam.read()
        
        if result:
            # Preprocess the image for YOLOv3
            image_size = image.shape[:2]
            # image_data = np.array(preprocess_image(image, (416, 416)), dtype=np.float32)
            image_data = np.array(letterbox_image(image, (416, 416)) / 255., dtype=np.float32)

            # Fetch data to DPU and trigger it
            input_data[0][...] = image_data.reshape(input_data[0].shape)
            job_id = dpu.execute_async(input_data, output_data)
            dpu.wait(job_id)

            # Retrieve the YOLOv3 output from DPU execution
            yolo_outputs = [output_data[0], output_data[1], output_data[2]]
            
            input_shape = np.shape(yolo_outputs[0])[1 : 3]
            input_shape = np.array(input_shape)*32
            
            # Extract people from the detections
            boxes, scores, classes = postprocess_boxes(yolo_outputs, image.shape[:2], input_shape, score_threshold=0.3)
            
            print("Result", scores, classes, boxes)


            # Create a dictionary for full image and sub-images (cropped people)
            img_dict = {
                "fullImage": process_image(image),
                "subImages": []
            }

            # Crop and process sub-images for detected people
            # Ensure bounding boxes are valid and within the image size
            for i, box in enumerate(boxes):
                x, y, w, h = map(int, box)

                # Ensure the coordinates are within the image dimensions
                x = max(0, x)
                y = max(0, y)
                w = min(w, image.shape[1] - x)  # Ensure the width stays within bounds
                h = min(h, image.shape[0] - y)  # Ensure the height stays within bounds

                # Crop the valid region from the image
                cropped_person = image[y:y+h, x:x+w]

                img_dict["subImages"].append({
                    "image": process_image(cropped_person),
                    "boundingBox": (x, y, w, h)
                })

            # Encode the image in JSON format as a string
            payload = json.dumps(img_dict)
            print("Sending image...")

            # Send the Base64-encoded image string to the client
            await websocket.send(payload)
            print("Image sent successfully.")

# Start the WebSocket server on localhost:2424
async def main():
    address = "192.168.137.172"
    port = 2424
    
    print(f"WebSocket server started on ws://{address}:{port}")
    server = await websockets.serve(send_image, address, port)
    
    await server.wait_closed()

asyncio.run(main())
