/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-webgpu';

import * as mpPose from '@mediapipe/pose';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
import * as tf from '@tensorflow/tfjs-core';

tfjsWasm.setWasmPaths(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
        tfjsWasm.version_wasm}/dist/`);

import * as posedetection from '@tensorflow-models/pose-detection';

import {Camera} from './websocket';
import {RendererCanvas2d} from './renderer_canvas2d';
import {setupDatGui} from './option_panel';
import {STATE} from './params';
import {setupStats} from './stats_panel';
import {setBackendAndEnvFlags} from './util';

let detector, camera, stats;
let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;
let rafId;
let renderer = null;
let useGpuRenderer = false;

async function createDetector() {
  switch (STATE.model) {
    case posedetection.SupportedModels.PoseNet:
      return posedetection.createDetector(STATE.model, {
        quantBytes: 4,
        architecture: 'MobileNetV1',
        outputStride: 16,
        inputResolution: {width: 500, height: 500},
        multiplier: 0.75
      });
    case posedetection.SupportedModels.BlazePose:
      const runtime = STATE.backend.split('-')[0];
      if (runtime === 'mediapipe') {
        return posedetection.createDetector(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
          solutionPath:
              `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${mpPose.VERSION}`
        });
      } else if (runtime === 'tfjs') {
        return posedetection.createDetector(
            STATE.model, {runtime, modelType: STATE.modelConfig.type});
      }
    case posedetection.SupportedModels.MoveNet:
      let modelType;
      if (STATE.modelConfig.type == 'lightning') {
        modelType = posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING;
      } else if (STATE.modelConfig.type == 'thunder') {
        modelType = posedetection.movenet.modelType.SINGLEPOSE_THUNDER;
      } else if (STATE.modelConfig.type == 'multipose') {
        modelType = posedetection.movenet.modelType.MULTIPOSE_LIGHTNING;
      }
      const modelConfig = {modelType};

      if (STATE.modelConfig.customModel !== '') {
        modelConfig.modelUrl = STATE.modelConfig.customModel;
      }
      if (STATE.modelConfig.type === 'multipose') {
        modelConfig.enableTracking = STATE.modelConfig.enableTracking;
      }
      return posedetection.createDetector(STATE.model, modelConfig);
  }
}

async function checkGuiUpdate() {
  if (STATE.isTargetFPSChanged || STATE.isSizeOptionChanged) {
    camera = await Camera.setupCamera(STATE.websocket);
    STATE.isTargetFPSChanged = false;
    STATE.isSizeOptionChanged = false;
  }

  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true;

    window.cancelAnimationFrame(rafId);

    if (detector != null) {
      detector.dispose();
    }

    if (STATE.isFlagChanged || STATE.isBackendChanged) {
      await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    }

    try {
      detector = await createDetector(STATE.model);
    } catch (error) {
      detector = null;
      alert(error);
    }

    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }
}

function beginEstimatePosesStats() {
  startInferenceTime = (performance || Date).now();
}

function endEstimatePosesStats() {
  const endInferenceTime = (performance || Date).now();
  inferenceTimeSum += endInferenceTime - startInferenceTime;
  ++numInferences;

  const panelUpdateMilliseconds = 1000;
  if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
    const averageInferenceTime = inferenceTimeSum / numInferences;
    inferenceTimeSum = 0;
    numInferences = 0;
    stats.customFpsPanel.update(
        1000.0 / averageInferenceTime, 120 /* maxValue */);
    lastPanelUpdate = endInferenceTime;
  }
}

async function renderResult() {
  await new Promise((resolve) => {
    camera.onframe = () => {
      resolve();
    };
  });

  let poses = [];
  let canvasInfo = null;

  // Detector can be null if initialization failed (for example when loading
  // from a URL that does not exist).
  if (detector != null) {
    // FPS only counts the time it takes to finish estimatePoses.
    beginEstimatePosesStats();

    if (useGpuRenderer && STATE.model !== 'PoseNet') {
      throw new Error('Only PoseNet supports GPU renderer!');
    }
    // Detectors can throw errors, for example when using custom URLs that
    // contain a model that doesn't provide the expected output.
    try {
      for (let i = 0; i < camera.subframes.images.length; i++) {
        const subImage = camera.subframes.images[i];
        const pose = await detector.estimatePoses(subImage, {
          maxPoses: STATE.modelConfig.maxPoses,
          flipHorizontal: false,
        });
        if (pose.length > 1) {
          // Push all poses in the subimage to the poses array and transform the keypoints to the original image coordinates
          pose.map((p) => {
            p.keypoints.map((keypoint) => {
              keypoint.x += camera.subframes.boundingBoxes[i][0];
              keypoint.y += camera.subframes.boundingBoxes[i][1];
            });
            poses.push(p);
            
          });
        }
      }
      // console.log(poses);
          
    } catch (error) {
      detector.dispose();
      detector = null;
      alert(error);
    }

    extractOrientation(poses);

    calculateAttentionScore(poses);

    filterSkeletons(poses);

    endEstimatePosesStats();
  }
  const rendererParams = useGpuRenderer ?
      [camera, poses, canvasInfo, STATE.modelConfig.scoreThreshold] :
      [camera, poses, STATE.isModelChanged];
  renderer.draw(rendererParams);
}

function filterSkeletons(poses) {
  // Apply to each pose in poses
  poses.map((pose) => {
    // Filter out the keypoints that are not on the head
    pose.keypoints = pose.keypoints.filter((keypoint) => {
      return keypoint.name === 'nose' || keypoint.name === 'left_eye' || keypoint.name === 'right_eye' || keypoint.name === 'left_ear' || keypoint.name === 'right_ear';
    });
  });
}


function extractOrientation(poses) {
  // Apply to each pose in poses
  poses.map((pose) => {
    // Estmate the angle ther person is looking at by the nose and the eyes.
    const nose = pose.keypoints.find((keypoint) => keypoint.name === 'nose');
    const leftEye = pose.keypoints.find((keypoint) => keypoint.name === 'left_eye');
    const rightEye = pose.keypoints.find((keypoint) => keypoint.name === 'right_eye');
    const leftEar = pose.keypoints.find((keypoint) => keypoint.name === 'left_ear');
    const rightEar = pose.keypoints.find((keypoint) => keypoint.name === 'right_ear');

    // Calculate the horizontal distance between the left and right eyes
    const eyeDistanceX = rightEye.x - leftEye.x;
    const eyeDistanceY = rightEye.y - leftEye.y;
    const eyeDistance = Math.sqrt(eyeDistanceX ** 2 + eyeDistanceY ** 2);

    // Calculate the horizontal distance between the left and right ears
    const earDistanceX = rightEar.x - leftEar.x;
    const earDistanceY = rightEar.y - leftEar.y;
    const earDistance = Math.sqrt(earDistanceX ** 2 + earDistanceY ** 2);

    // Calculate the midpoint between the eyes
    const eyeMidPoint = {
        x: (leftEye.x + rightEye.x) / 2,
        y: (leftEye.y + rightEye.y) / 2,
    };

    const earMidPoint = {
        x: (leftEar.x + rightEar.x) / 2,
        y: (leftEar.y + rightEar.y) / 2,
    };

    // Calculate the vertical distance between the nose and the midpoint of the eyes
    const tiltY = nose.y - earMidPoint.y;

    // Calculate head tilt (up-down rotation)
    const tilt = Math.atan2(tiltY, eyeDistance) * (180 / Math.PI); // In degrees

    // Calculate head yaw (left-right rotation)
    const yaw = Math.atan2(eyeMidPoint.x - nose.x, eyeDistance) * (180 / Math.PI); // In degrees

    // console.log(`Head Yaw (left-right): ${yaw} degrees`);
    // console.log(`Head Tilt (up-down): ${tilt} degrees`);
    pose.orientation = {
      yaw: yaw,
      tilt: tilt
    };
  });
}

// Reference direction (e.g., facing forward with no tilt)
const referenceYaw = 0; // Facing straight ahead
const referenceTilt = 10; // No upward or downward tilt

// Calculate the deviation of a person's head orientation from the reference
function calculateAttentionScore(poses) {
  poses.map((pose) => {

    const yaw = pose.orientation.yaw;
    const tilt = pose.orientation.tilt;

    // Maximum allowed deviation for full attention (tuned based on real-world data)
    const maxYawDeviation = 40;  // Example: 45 degrees allowed yaw deviation
    const maxTiltDeviation = 30; // Example: 30 degrees allowed tilt deviation

    // Calculate the yaw and tilt deviation from the reference direction
    const yawDeviation = Math.abs(yaw - referenceYaw);
    const tiltDeviation = Math.abs(tilt - referenceTilt);

    // Normalize the deviations to get a score between 0 and 1
    const yawScore = Math.max(0, 1 - yawDeviation / maxYawDeviation);
    const tiltScore = Math.max(0, 1 - tiltDeviation / maxTiltDeviation);

    // Final attention score: combine yaw and tilt scores (weighted average)
    const attentionScore = (yawScore + tiltScore) / 2;

    pose.attentionScore = attentionScore * 100; // Convert to percentage
  });
}


async function renderPrediction() {
  await checkGuiUpdate();

  if (!STATE.isModelChanged) {
    await renderResult();
  }

  rafId = requestAnimationFrame(renderPrediction);
};

async function app() {
  // Gui content will change depending on which model is in the query string.
  const urlParams = new URLSearchParams(window.location.search);
  // if (!urlParams.has('model')) {
  //   alert('Cannot find model in the query string.');
  //   return;
  // }
  await setupDatGui(urlParams);

  stats = setupStats();
  const isWebGPU = STATE.backend === 'tfjs-webgpu';
  const importVideo = (urlParams.get('importVideo') === 'true') && isWebGPU;

  camera = await Camera.setup(STATE.websocket);

  await setBackendAndEnvFlags(STATE.flags, STATE.backend);
  await tf.ready();
  detector = await createDetector();
  const canvas_frame = document.getElementById('frame');
  canvas_frame.width = camera.frame.width;
  canvas_frame.height = camera.frame.height;

  const canvas_subframes = document.getElementById('subframes');
  canvas_subframes.width = camera.subframes.width;
  canvas_subframes.height = camera.subframes.height;

  renderer = new RendererCanvas2d(canvas_frame, canvas_subframes);

  renderPrediction();
};

app();

if (useGpuRenderer) {
  renderer.dispose();
}
