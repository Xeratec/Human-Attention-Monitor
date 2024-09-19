import * as params from './params';
var W3CWebSocket = require('websocket').w3cwebsocket;


export class Camera {
  constructor() {
    this.frame = document.getElementById("frame");
    this.subframes = document.getElementById("subframes");
    this.subframes.images = []; // Array to store images
    this.subframes.boundingBoxes = []; // Array to store bounding boxes
    this.webSocket = null; // WebSocket for receiving images
  }

  /**
   * Setup a WebSocket connection and receive image stream.
   * @param {string} wsUrl - The WebSocket URL to connect to.
   */
  static async setup(websocketParam) {
    const camera = new Camera();

    console.log(
      "Setting up camera with WebSocket fetching image from" +
        websocketParam.wsUrl
    );

    // Initialize WebSocket and close existing connections
    camera.webSocket = new W3CWebSocket(websocketParam.wsUrl);

    camera.webSocket.onopen = () => {
      console.log("WebSocket connection established.");
    };

    camera.webSocket.onmessage = (event) => {
      // Decode message as json object
      const message = JSON.parse(event.data);
      camera.displayImage(message);
    };

    camera.webSocket.onclose = () => {
      console.log("WebSocket connection closed.");
    };

    return camera;
  }

  /**
   * Display an image on the canvas from Base64 data.
   * @param {string} base64Data - Base64-encoded image data.
   */
  displayImage(message) {
    // console.log(message);
    const img = new Image();
    img.src = "data:image/jpeg;base64," + message.fullImage;
    // Resize the canvas to match the image size
    this.frame = img;

    // Delete old images
    this.subframes.images = [];
    this.subframes.boundingBoxes = [];

    for (let img of message.subImages) {
      const subImg = new Image();
      subImg.src = "data:image/jpeg;base64," + img.image;
      this.subframes.images.push(subImg);
      this.subframes.boundingBoxes.push(img.boundingBox);
    }

    // Set video to viewport size

    this.frame.width = img.width;
    this.frame.height = img.height;

    // Sum the width of all subframes
    this.subframes.width = this.subframes.images.reduce(
      (total, img) => total + img.width,
      0
    );
    // Set the height to maximum of all subframes
    this.subframes.height = this.subframes.images.reduce(
      (max, img) => Math.max(max, img.height),
      0
    );

    const canvasContainer = document.querySelector(".canvas-wrapper");
    canvasContainer.style = `width: ${this.frame.width}px; height: ${this.frame.height}px`;
    if (this.onframe != undefined) {
      this.onframe();
    }
  }
}
