// Copyright 2026 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.mediapipe.apps.faceheadroigpu;

import android.os.Bundle;
import android.util.Log;
import android.view.ViewGroup;
import com.google.mediapipe.apps.basic.R;
import com.google.mediapipe.formats.proto.ClassificationProto.Classification;
import com.google.mediapipe.formats.proto.ClassificationProto.ClassificationList;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList;
import com.google.mediapipe.formats.proto.RectProto.NormalizedRect;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.framework.ProtoUtil;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/** Android app for manual Head ROI face landmarks + blendshapes. */
public class MainActivity extends com.google.mediapipe.apps.basic.MainActivity {
  private static final String TAG = "FaceHeadRoi";

  private static final String INPUT_ROI_STREAM_NAME = "roi_rect";
  private static final String OUTPUT_LANDMARKS_STREAM_NAME = "multi_face_landmarks";
  private static final String OUTPUT_BLENDSHAPES_STREAM_NAME = "blendshapes";

  private final AtomicReference<NormalizedRect> currentRoi =
      new AtomicReference<>(
          NormalizedRect.newBuilder()
              .setXCenter(0.5f)
              .setYCenter(0.5f)
              .setWidth(0.6f)
              .setHeight(0.7f)
              .setRotation(0.0f)
              .build());

  private final Object fpsLock = new Object();
  private long lastLandmarkTimestamp = Long.MIN_VALUE;
  private double smoothedInferFps = 0.0;

  // Debug counters
  private final AtomicInteger roiSentCount = new AtomicInteger(0);
  private final AtomicInteger landmarkCallbackCount = new AtomicInteger(0);
  private final AtomicLong lastLandmarkCallbackMs = new AtomicLong(0);

  private HeadRoiOverlayView overlayView;

  static {
    ProtoUtil.registerTypeName(NormalizedRect.class, "mediapipe.NormalizedRect");
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    overlayView = new HeadRoiOverlayView(this);
    overlayView.setRoiChangeListener(currentRoi::set);

    ViewGroup viewGroup = findViewById(R.id.preview_display_layout);
    viewGroup.addView(
        overlayView,
        new ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));

    processor.addPacketCallback(
        OUTPUT_LANDMARKS_STREAM_NAME,
        (packet) -> {
          List<NormalizedLandmarkList> multiFaceLandmarks =
              PacketGetter.getProtoVector(packet, NormalizedLandmarkList.parser());
          updateInferenceFps(packet.getTimestamp());
          int cbCount = landmarkCallbackCount.incrementAndGet();
          lastLandmarkCallbackMs.set(System.currentTimeMillis());
          int numFaces = multiFaceLandmarks.size();
          int numLandmarks = numFaces > 0 ? multiFaceLandmarks.get(0).getLandmarkCount() : 0;
          // Log every callback so we see if it fires at all, then throttle to every 30.
          if (cbCount <= 5 || cbCount % 30 == 0) {
            Log.d(TAG, "landmarks_cb #" + cbCount
                + " ts=" + packet.getTimestamp()
                + " faces=" + numFaces
                + " landmarks[0]=" + numLandmarks);
          }
          NormalizedLandmarkList landmarks =
              multiFaceLandmarks.isEmpty()
                  ? NormalizedLandmarkList.getDefaultInstance()
                  : multiFaceLandmarks.get(0);
          runOnUiThread(
              () -> {
                overlayView.setLandmarks(landmarks);
                overlayView.setInferenceFps(smoothedInferFps);
              });
        });

    processor.addPacketCallback(
        OUTPUT_BLENDSHAPES_STREAM_NAME,
        (packet) -> {
          List<ClassificationList> lists =
              PacketGetter.getProtoVector(packet, ClassificationList.parser());
          List<Classification> classifications =
              lists.isEmpty() ? Collections.emptyList() : lists.get(0).getClassificationList();
          runOnUiThread(() -> overlayView.setBlendshapes(classifications));
        });

    processor.setOnWillAddFrameListener(
        (timestamp) -> {
          Packet roiPacket = null;
          try {
            NormalizedRect roi = currentRoi.get();
            roiPacket = processor.getPacketCreator().createProto(roi);
            processor.getGraph().addPacketToInputStream(INPUT_ROI_STREAM_NAME, roiPacket, timestamp);
            int sent = roiSentCount.incrementAndGet();
            // Log first 5 sends and then every 60 frames.
            if (sent <= 5 || sent % 60 == 0) {
              Log.d(TAG, "roi_sent #" + sent
                  + " ts=" + timestamp
                  + " cx=" + roi.getXCenter()
                  + " cy=" + roi.getYCenter()
                  + " w=" + roi.getWidth()
                  + " h=" + roi.getHeight());
              // Warn if landmark callback hasn't fired for > 3 s.
              long lastCbMs = lastLandmarkCallbackMs.get();
              if (lastCbMs == 0) {
                Log.w(TAG, "landmarks_cb has NEVER fired after " + sent + " frames");
              } else {
                long silenceMs = System.currentTimeMillis() - lastCbMs;
                if (silenceMs > 3000) {
                  Log.w(TAG, "landmarks_cb silent for " + silenceMs + " ms");
                }
              }
            }
          } catch (RuntimeException e) {
            Log.e(TAG, "Failed to send Head ROI packet.", e);
          } finally {
            if (roiPacket != null) {
              roiPacket.release();
            }
          }
        });
  }

  private void updateInferenceFps(long packetTimestamp) {
    synchronized (fpsLock) {
      if (lastLandmarkTimestamp != Long.MIN_VALUE && packetTimestamp > lastLandmarkTimestamp) {
        double fps = 1_000_000.0 / (packetTimestamp - lastLandmarkTimestamp);
        smoothedInferFps = smoothedInferFps == 0.0 ? fps : smoothedInferFps * 0.9 + fps * 0.1;
      }
      lastLandmarkTimestamp = packetTimestamp;
    }
  }
}
