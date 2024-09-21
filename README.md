# Human Attention Monitor

> **HEADS UP: We know when you are zoomed out**

This project was developed as part of the Creative Lab of the [CPS Summer School](https://www.cpsschool.eu/) 24 which was held in Alghero (Italy), from 16th to 20th September 2024 and received the best creative lab project award.

## Overview

The objective of this project is to detect the attention level of an audience to provide feedback to a lecturer for improvement and real-time information on when a break is needed. The project demonstrates a Cyber-Physical System (CPS) where person detection is handled by an edge platform using a reconfigurable computing platform (AMD-Xilinx Kira KV260 Vision AI Starter Kit).

Key features include:
- Person detection handled by the FPGA unsing YoloV3 and PYNQ
- Data transmission via WebSocket to a JavaScript interface.
- Pose estimation for each person using MoveNet.
- Calculation of an attention score based on head direction.
- Display of average attention in numerical format and as a bar plot.

## Installation

To use this project, you need to install NodeJS and Yarn. After that, you can clone the repository and install the dependencies.

To install [NodeJS](https://nodejs.org/) and Yarn, you can use the following commands:

```bash
###### Install NodeJS ########
$> sudo apt install nodejs # Ubuntu
$> sudo pacman -S nodejs # Arch Linux
$> sudo dnf install nodejs # Fedora
$> brew install node
# or download the installer from https://nodejs.org/

###### Install Yarn ########
$> npm install --global yarn
```

After installing NodeJS and Yarn, you can clone the repository and install the dependencies:

```bash
# Clone the repository
$> git clone https://github.com/Xeratec/Human-Attention-Monitor
$> cd Human-Attention-Monitor

# Install the dependencies
$> yarn
```

## Usage

Two different modes are available to run the project: with and without the FPGA. The first mode uses the Kira KV260 Vision AI Starter Kit to acquire the image, run the people detection, and send the data to the web server. The second mode uses the camera of your device to acquire the image and run the MoveNet pose estimation without people detection.

### With FPGA
To start the people detection on the FPGA, you need to run the following command on the board:

```bash
# Activate the environment
source <path-to-pynq-setup-script>

# Start the people detection on the FPGA
python scripts/app_yolov3.py
```

Alternatively, you can use the face detection on using OpenCV and a haarcascade classifier:

```bash
python scripts/app_face.py
```

Next, you can start the web server to display the attention score:

```bash
# Start the web server
$> yarn watch
```

### Standalone

If you want to run the project without the FPGA, you can use the following command:

```bash
# Start the web server
$> yarn watch-standalone
```

This will use the camera of your device and run MoveNet on the whole image.

## Known Issues
It seems like there is a bug in the people detection leading to wrong bounding boxes. We recommend to reimplement most of the `app_yolov3.py` script based on the example provided by the CPS team.

## Authors
- **Philip Wiese** (ETH Zurich)
- **Alessandro Monni** (Università degli Studi di Sassari)
- **Daniele Nicoletti** (Università degli Studi di Verona)
- **Leonardo Picchiami** (Sapienza Università di Roma)
