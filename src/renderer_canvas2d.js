/**
 * @license
 * Copyright 2023 Google LLC.
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
import * as posedetection from '@tensorflow-models/pose-detection';
import * as scatter from 'scatter-gl';

import * as params from './params';

// These anchor points allow the pose pointcloud to resize according to its
// position in the input.
const ANCHOR_POINTS = [[0, 0, 0], [0, 1, 0], [-1, 0, 0], [-1, -1, 0]];

// #ffffff - White
// #800000 - Maroon
// #469990 - Malachite
// #e6194b - Crimson
// #42d4f4 - Picton Blue
// #fabed4 - Cupid
// #aaffc3 - Mint Green
// #9a6324 - Kumera
// #000075 - Navy Blue
// #f58231 - Jaffa
// #4363d8 - Royal Blue
// #ffd8b1 - Caramel
// #dcbeff - Mauve
// #808000 - Olive
// #ffe119 - Candlelight
// #911eb4 - Seance
// #bfef45 - Inchworm
// #f032e6 - Razzle Dazzle Rose
// #3cb44b - Chateau Green
// #a9a9a9 - Silver Chalice
const COLOR_PALETTE = [
  '#ffffff', '#800000', '#469990', '#e6194b', '#42d4f4', '#fabed4', '#aaffc3',
  '#9a6324', '#000075', '#f58231', '#4363d8', '#ffd8b1', '#dcbeff', '#808000',
  '#ffe119', '#911eb4', '#bfef45', '#f032e6', '#3cb44b', '#a9a9a9'
];
export class RendererCanvas2d {
  constructor(canvas_frame, canvas_subframes) {
    this.canvas_frame = canvas_frame.getContext('2d');
    this.canvas_subframes = canvas_subframes.getContext('2d');
    this.canvasContainer = document.querySelector(".canvas-wrapper");
    this.videoWidth = canvas_frame.width;
    this.videoHeight = canvas_frame.height;
    this.filteredAttentionScore = 0;
  }

  flip(videoWidth, videoHeight) {
    // Because the image from camera is mirrored, need to flip horizontally.
    this.canvas_frame.translate(videoWidth, 0);
    this.canvas_frame.scale(-1, 1);
  }

  draw(rendererParams) {
    const [camera, poses, isModelChanged] = rendererParams;
    this.drawCtx(camera);

    // The null check makes sure the UI is not in the middle of changing to a
    // different model. If during model change, the result is from an old model,
    // which shouldn't be rendered.

    if (poses && poses.length > 0 && !isModelChanged) {
      this.drawResults(poses);
    } else {
      this.drawGlobalAttention([])
    }
  }

  drawCtx(camera) {
    this.canvas_frame.drawImage(
      camera.frame,
      0,
      0,
      camera.frame.width,
      camera.frame.height
    );

    // Draw subimages below the main image in a row with 10px spacing

    for (let i = 0; i < camera.subframes.images.length; i++) {
      const subImg = camera.subframes.images[i];
      this.canvas_subframes.drawImage(
        subImg,
        i * (subImg.width + 0),
        0,
        subImg.width,
        subImg.height
      );

      // Draw bounding boxes in original image saved in subImg.boundingBox
      this.canvas_frame.strokeStyle = "red";
      this.canvas_frame.lineWidth = 2;
      this.canvas_frame.strokeRect(
        camera.subframes.boundingBoxes[i][0],
        camera.subframes.boundingBoxes[i][1],
        camera.subframes.boundingBoxes[i][2] -
          camera.subframes.boundingBoxes[i][0],
        camera.subframes.boundingBoxes[i][3] -
          camera.subframes.boundingBoxes[i][1]
      );
    }
  }

  clearCtx() {
    this.canvas_frame.clearRect(0, 0, this.videoWidth, this.videoHeight);
  }

  filterAttntionScore(poses) {
    // Calculate the average attention score across all poses
    const totalAttentionScore = poses.reduce((total, pose) => total + pose.attentionScore, 0);
    const averageAttentionScore =  totalAttentionScore / poses.length;

    const filterConstant = 0.95;
    this.filteredAttentionScore = filterConstant * this.filteredAttentionScore + (1 - filterConstant) * averageAttentionScore
    return this.filteredAttentionScore
  }

  /**
   * Draw the keypoints and skeleton on the video.
   * @param poses A list of poses to render.
   */
  drawResults(poses) {
    for (const pose of poses) {
      this.drawResult(pose);
    }
    this.filterAttntionScore(poses);
    this.drawGlobalAttention(poses);
  }

  drawGlobalAttention(poses) {
    const averageAttentionScore = this.filteredAttentionScore;
    this.canvas_frame.font = '20px Arial';
    this.canvas_frame.fillStyle = 'red';
    // this.ctx.fillText(`Average Attention: ${averageAttentionScore.toFixed(0)} %`, 10, 30);

    // Define bar dimensions
    const barY = this.videoHeight - 30;
    const barHeight = 20;

    const barWidth = this.videoWidth * 0.8;
    const barX = (this.videoWidth - barWidth) / 2;

    // Draw the bar outline
    // Draw background of the bar (gray)
    this.canvas_frame.fillStyle = '#ccc';
    this.canvas_frame.fillRect(barX, barY, barWidth, barHeight);

    if (poses == undefined) {
      return;
    }
    // Draw attention level (green, proportional to average attention)
    const attentionBarWidth = barWidth * (averageAttentionScore / 100);

    // calcualte color based on attention score form red to green
    const r = Math.floor(255 * (1 - averageAttentionScore / 100));
    const g = Math.floor(255 * (averageAttentionScore / 100));
    this.canvas_frame.fillStyle = `rgb(${r}, ${g}, 0)`;
    this.canvas_frame.fillRect(barX, barY, attentionBarWidth, barHeight);

    // // Draw a border around the bar
    // this.ctx.strokeStyle = 'black';
    // this.ctx.strokeRect(barX, barY, barWidth, barHeight);
  }

  /**
   * Draw the keypoints and skeleton on the video.
   * @param pose A pose with keypoints to render.
   */
  drawResult(pose) {
    if (pose.keypoints != null) {
      this.drawKeypoints(pose.keypoints);
      this.drawSkeleton(pose.keypoints, pose.id);
      this.drawAttention(pose);
    }
  }

  /**
   * Draw the keypoints on the video.
   * @param keypoints A list of keypoints.
   */
  drawKeypoints(keypoints) {
    const keypointInd =
        posedetection.util.getKeypointIndexBySide(params.STATE.model);
    this.canvas_frame.fillStyle = 'Red';
    this.canvas_frame.strokeStyle = 'White';
    this.canvas_frame.lineWidth = params.DEFAULT_LINE_WIDTH;

    for (const i of keypointInd.middle) {
      this.drawKeypoint(keypoints[i]);
    }

    this.canvas_frame.fillStyle = 'Green';
    for (const i of keypointInd.left) {
      this.drawKeypoint(keypoints[i]);
    }

    this.canvas_frame.fillStyle = 'Orange';
    for (const i of keypointInd.right) {
      this.drawKeypoint(keypoints[i]);
    }
  }

  drawKeypoint(keypoint) {
    // Check if keypoint is undefined
    if (keypoint == undefined) {
      return;
    }

    // If score is null, just show the keypoint.
    const score = keypoint.score != null ? keypoint.score : 1;
    const scoreThreshold = params.STATE.modelConfig.scoreThreshold || 0;

    if (score >= scoreThreshold) {
      const circle = new Path2D();
      circle.arc(keypoint.x, keypoint.y, params.DEFAULT_RADIUS, 0, 2 * Math.PI);
      this.canvas_frame.fill(circle);
      this.canvas_frame.stroke(circle);
    }
  }

  drawAttention(pose) {
    const orientation = pose.orientation;
    const keypoints = pose.keypoints;

    // Draw the angle near the nose
    const nose = keypoints.find((keypoint) => keypoint.name === 'nose');

    // Orientation contains yaw and tilt
    const yaw = orientation.yaw;
    const tilt = orientation.tilt;

    // Draw yaw and mirror it
    this.canvas_frame.font = '20px Arial';
    this.canvas_frame.fillStyle = 'red';
    // this.ctx.fillText(`Yaw: ${yaw.toFixed(0)} degrees`, nose.x, nose.y);
    // this.ctx.fillText(`Tilt: ${tilt.toFixed(0)} degrees`, nose.x, nose.y +
    //     30);
    // this.ctx.fillText(`Attention: ${pose.attentionScore.toFixed(0)} %`, nose.x, nose.y + 60);
  }
  /**
   * Draw the skeleton of a body on the video.
   * @param keypoints A list of keypoints.
   */
  drawSkeleton(keypoints, poseId) {
    // Each poseId is mapped to a color in the color palette.
    const color = params.STATE.modelConfig.enableTracking && poseId != null ?
        COLOR_PALETTE[poseId % 20] :
        'White';
    this.canvas_frame.fillStyle = color;
    this.canvas_frame.strokeStyle = color;
    this.canvas_frame.lineWidth = params.DEFAULT_LINE_WIDTH;

    posedetection.util.getAdjacentPairs(params.STATE.model).forEach(([
                                                                      i, j
                                                                    ]) => {
      const kp1 = keypoints[i];
      const kp2 = keypoints[j];

      if (kp1 == undefined || kp2 == undefined) {
        return;
      }

      // If score is null, just show the keypoint.
      const score1 = kp1.score != null ? kp1.score : 1;
      const score2 = kp2.score != null ? kp2.score : 1;
      const scoreThreshold = params.STATE.modelConfig.scoreThreshold || 0;

      if (score1 >= scoreThreshold && score2 >= scoreThreshold) {
        this.canvas_frame.beginPath();
        this.canvas_frame.moveTo(kp1.x, kp1.y);
        this.canvas_frame.lineTo(kp2.x, kp2.y);
        this.canvas_frame.stroke();
      }
    });
  }
}