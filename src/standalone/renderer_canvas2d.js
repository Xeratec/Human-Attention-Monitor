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

import * as params from '../params';

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
  constructor(canvas) {
    this.ctx = canvas.getContext('2d');
    this.scatterGLEl = document.querySelector('#scatter-gl-container');
    this.scatterGL = new scatter.ScatterGL(this.scatterGLEl, {
      'rotateOnStart': true,
      'selectEnabled': false,
      'styles': {polyline: {defaultOpacity: 1, deselectedOpacity: 1}}
    });
    this.scatterGLHasInitialized = false;
    this.videoWidth = canvas.width;
    this.videoHeight = canvas.height;
    this.filteredAttentionScore = 0;
    // this.flip(this.videoWidth, this.videoHeight);
  }

  flip(videoWidth, videoHeight) {
    // Because the image from camera is mirrored, need to flip horizontally.
    this.ctx.translate(videoWidth, 0);
    this.ctx.scale(-1, 1);

    this.scatterGLEl.style =
        `width: ${videoWidth}px; height: ${videoHeight}px;`;
    this.scatterGL.resize();

    this.scatterGLEl.style.display =
        params.STATE.modelConfig.render3D ? 'inline-block' : 'none';
  }

  draw(rendererParams) {
    const [video, poses, isModelChanged] = rendererParams;
    this.drawCtx(video);

    // The null check makes sure the UI is not in the middle of changing to a
    // different model. If during model change, the result is from an old model,
    // which shouldn't be rendered.

    if (poses && poses.length > 0 && !isModelChanged) {
      this.drawResults(poses);
    } else {
      this.drawGlobalAttention([])
    }
  }

  drawCtx(video) {
    // this.ctx.save();
    // this.ctx.scale(-1, 1);
    // this.ctx.drawImage(video, 0, 0, -this.videoWidth, this.videoHeight);
    // this.ctx.restore();
    this.ctx.drawImage(video, 0, 0, this.videoWidth, this.videoHeight);
  }

  clearCtx() {
    this.ctx.clearRect(0, 0, this.videoWidth, this.videoHeight);
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
    this.ctx.font = '20px Arial';
    this.ctx.fillStyle = 'red';
    // this.ctx.fillText(`Average Attention: ${averageAttentionScore.toFixed(0)} %`, 10, 30);

    // Define bar dimensions
    const barY = this.videoHeight - 30;
    const barHeight = 20;

    const barWidth = this.videoWidth * 0.8;
    const barX = (this.videoWidth - barWidth) / 2;

    // Draw the bar outline
    // Draw background of the bar (gray)
    this.ctx.fillStyle = '#ccc';
    this.ctx.fillRect(barX, barY, barWidth, barHeight);

    if (poses == undefined) {
      return;
    }
    // Draw attention level (green, proportional to average attention)
    const attentionBarWidth = barWidth * (averageAttentionScore / 100);

    // calcualte color based on attention score form red to green
    const r = Math.floor(255 * (1 - averageAttentionScore / 100));
    const g = Math.floor(255 * (averageAttentionScore / 100));
    this.ctx.fillStyle = `rgb(${r}, ${g}, 0)`;
    this.ctx.fillRect(barX, barY, attentionBarWidth, barHeight);

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
    if (pose.keypoints3D != null && params.STATE.modelConfig.render3D) {
      this.drawKeypoints3D(pose.keypoints3D);
    }
  }

  /**
   * Draw the keypoints on the video.
   * @param keypoints A list of keypoints.
   */
  drawKeypoints(keypoints) {
    const keypointInd =
        posedetection.util.getKeypointIndexBySide(params.STATE.model);
    this.ctx.fillStyle = 'Red';
    this.ctx.strokeStyle = 'White';
    this.ctx.lineWidth = params.DEFAULT_LINE_WIDTH;

    for (const i of keypointInd.middle) {
      this.drawKeypoint(keypoints[i]);
    }

    this.ctx.fillStyle = 'Green';
    for (const i of keypointInd.left) {
      this.drawKeypoint(keypoints[i]);
    }

    this.ctx.fillStyle = 'Orange';
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
      this.ctx.fill(circle);
      this.ctx.stroke(circle);
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
    this.ctx.font = '20px Arial';
    this.ctx.fillStyle = 'red';
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
    this.ctx.fillStyle = color;
    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = params.DEFAULT_LINE_WIDTH;

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
        this.ctx.beginPath();
        this.ctx.moveTo(kp1.x, kp1.y);
        this.ctx.lineTo(kp2.x, kp2.y);
        this.ctx.stroke();
      }
    });
  }

  drawKeypoints3D(keypoints) {
    const scoreThreshold = params.STATE.modelConfig.scoreThreshold || 0;
    const pointsData =
        keypoints.map(keypoint => ([-keypoint.x, -keypoint.y, -keypoint.z]));

    const dataset =
        new scatter.ScatterGL.Dataset([...pointsData, ...ANCHOR_POINTS]);

    const keypointInd =
        posedetection.util.getKeypointIndexBySide(params.STATE.model);
    this.scatterGL.setPointColorer((i) => {
      if (keypoints[i] == null || keypoints[i].score < scoreThreshold) {
        // hide anchor points and low-confident points.
        return '#ffffff';
      }
      if (i === 0) {
        return '#ff0000' /* Red */;
      }
      if (keypointInd.left.indexOf(i) > -1) {
        return '#00ff00' /* Green */;
      }
      if (keypointInd.right.indexOf(i) > -1) {
        return '#ffa500' /* Orange */;
      }
    });

    if (!this.scatterGLHasInitialized) {
      this.scatterGL.render(dataset);
    } else {
      this.scatterGL.updateDataset(dataset);
    }
    const connections = posedetection.util.getAdjacentPairs(params.STATE.model);
    const sequences = connections.map(pair => ({indices: pair}));
    this.scatterGL.setSequences(sequences);
    this.scatterGLHasInitialized = true;
  }
}
