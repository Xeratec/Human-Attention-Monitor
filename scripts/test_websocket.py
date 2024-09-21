import os
import cv2
import base64
import asyncio
import websockets
import json
import random
from pprint import pprint

cam = cv2.VideoCapture(0)

# Function to load, resize, and encode the image in Base64
def process_image(width = 800):
    # Capture the image from the webcam
    result, img = cam.read()

    h, w, _ = img.shape

    # Crop the image to a width while maintaining the aspect ratio
    img_resized = cv2.resize(img, (width, int(h * width / w)))

    # Save the resized image to a bytes buffer in JPEG format
    result, buffered = cv2.imencode('.jpg', img_resized)

    # Encode the image in Base64
    img_base64 = base64.b64encode(buffered).decode('utf-8')

    return img_resized, img_base64

# Function to extract n random sub-images from the full image
def extract_sub_images(img, n=2):

    # Get the width and height of the full image
    height, width, _ = img.shape

    # Initialize list to store sub-images and bounding boxes
    sub_images = []
    bounding_boxes = []

    # Extract n random sub-images from the full image
    for _ in range(n):
        # Generate random coordinates for the top-left corner of the sub-image
        # x = random.randint(0, width - 100)
        # y = random.randint(0, height - 100)
        x = 20
        y = 20
        width = int(width * 0.5)
        height = int(height * 0.8)

        # Crop the sub-image using the ROI
        sub_img = img[y:y + height, x:x + width]

        # Save the sub-image to a bytes buffer in JPEG format
        result, buffered = cv2.imencode('.jpg', sub_img)

        # Encode the sub-image in Base64
        sub_img_base64 = base64.b64encode(buffered).decode('utf-8')

        # Store the Base64-encoded sub-image in the list
        sub_images.append(sub_img_base64)
        bounding_boxes.append((x, y, x + width, y + height))

    return sub_images, bounding_boxes

# WebSocket server handler to send the image
async def send_image(websocket, path):
    # Send image every second
    while True:
        print("Client connected")

        # Process the image and get the Base64-encoded string
        img, img_base64 = process_image()

        # Extract n random sub-images from the full image
        sub_img_base64, boundingBoxes = extract_sub_images(img, n=1)

        # Assemble dictionary containing full image and sub-images
        img_dict = {
            "fullImage": img_base64,
            "subImages": []
        }

        for img, bbox in zip(sub_img_base64, boundingBoxes):
            img_dict["subImages"].append({
                "image": img,
                "boundingBox": bbox
            })

        # Encode image in JSON format as a string
        payload = json.dumps(img_dict)

        # Send the Base64-encoded image string to the client
        await websocket.send(payload)
        print("Image sent successfully.")

# Start the WebSocket server on localhost:2424
async def main():
    server = await websockets.serve(send_image, "localhost", 2424)
    print("WebSocket server started on ws://localhost:2424")

    await server.wait_closed()

# Run the WebSocket server
if __name__ == "__main__":
    asyncio.run(main())
