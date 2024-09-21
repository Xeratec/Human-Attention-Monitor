#import
import cv2
import matplotlib.pyplot as plt
import os
import base64
import asyncio
import websockets
from PIL import Image
from io import BytesIO
import json
import random
from pprint import pprint
import subprocess
import numpy as np

#----------------------------------------------------------------

# Function to encode the image in Base64 from an image variable
def process_image(img):
    
    result, buffered = cv2.imencode('.jpg', img)
    if result:
        # Encode the image in Base64
        img_base64 = base64.b64encode(buffered).decode('utf-8')
        return img_base64
    else:
        print("Error during image encoding")
        return None

#----------------------------------------------------------------

# WebSocket server handler to send the image
async def send_image(websocket, path):
    
    cam = cv2.VideoCapture(0)
    print("Client connected")
    
    # Send image every second
    while True:
        
         #-----------------------------TAKE-THE-IMAGE------------------------------------
        
        # orig_img_path = 'img/webcam.jpg'
        
        # Running a simple command
        # result = subprocess.Popen(['fswebcam', '--no-banner', '--save', f'{orig_img_path', '-d', '/dev/video0']).wait()  
        # result = subprocess.run(['./camera.sh'])
        result, image = cam.read()
        
        if result:
            # Carica l'immagine
            # image = cv2.imread(orig_img_path)

            # Converti l'immagine in formato RGB (perché OpenCV usa BGR di default)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            #----------------------------EXTRACT-FACES-----------------------------------------

            # Carica il classificatore pre-addestrato per il rilevamento dei volti
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            side_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

            # Converti l'immagine in scala di grigi
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Rileva i volti
            # Detect front faces
            front_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))

            # Detect side faces
            side_faces = side_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))

            # Merge both detections into a single list
            # Each entry is a tuple: (x, y, w, h)
            faces = np.concatenate((front_faces, side_faces), axis=0) if len(front_faces) > 0 and len(side_faces) > 0 else front_faces if len(front_faces) > 0 else side_faces

#             # Now `all_faces` contains both front and side face detections
#             for (x, y, w, h) in all_faces:
#                 # Here, you can process the bounding boxes as usual
#                 cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            height, width, channels = image.shape

            # Update the faces array with the expanded bounding boxes, including a 20% top expansion
            for i, (x, y, w, h) in enumerate(faces):
                # Calculate the new height (expand by 20% at the top, and keep the original height at the bottom)
                top_increase = int(0.5 * h)
                new_y = max(0, y - top_increase)  # Ensure the top boundary doesn't go out of the image
                new_h = 3*h + top_increase  # Adjust the height to include the top expansion
                new_h = min(new_h, height - new_y)  # Ensure the height doesn't exceed the image height

                # Expand the width symmetrically (left and right) but stay within image boundaries
                new_w = int(min(2 * w, width))  # Expanded width
                left_shift = (new_w - w) // 2  # Calculate how much to shift left

                # Ensure the left boundary doesn't go out of the image
                new_x = max(0, x - left_shift)
                # Ensure the right boundary doesn't exceed the image width
                new_w = min(new_w, width - new_x)

                # Overwrite the face bounding box with the new dimensions
                faces[i] = (new_x, new_y, new_w, new_h)

                # Optionally, draw the updated bounding boxes (if needed)
                # cv2.rectangle(image, (new_x, new_y), (new_x + new_w, new_y + new_h), (255, 0, 0), 2)

            #--insert-the-image-in-the-dictionary--

            img_dict = {
                "fullImage": process_image(image),
                "subImages": []
            }
            
            # jpg_original = base64.b64decode(img_dict["fullImage"])
            # jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            # img = cv2.imdecode(jpg_as_np, flags=1)
            # cv2.imwrite('img/output.jpg', img)

            #---------------------------------------------------------------------------------

            # Itera sui volti rilevati
            for i, (x, y, w, h) in enumerate(faces):

                # Croppa il volto dall'immagine originale
                cropped_face = image[y:y+h, x:x+w]

                # Salva il volto croppato come file JPEG
                # cv2.imwrite(f'img/face_{i}.jpeg', cropped_face)
                # print(f"Volto {i} salvato come 'face_{i}.jpg'")

                # Converte in RGB per visualizzazione con matplotlib
                cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)

                #--insert-the-bounding-boxes-in-the-dictionary--

                img_dict["subImages"].append({
                    "image": process_image(cropped_face_rgb),
                    "boundingBox": (int(x), int(y), int(w), int(h)) 
                })

            #--------------------------SEND-DICT-TO-WEB-SOCKET--------------------------------------


            # Encode image in JSON format as a string
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