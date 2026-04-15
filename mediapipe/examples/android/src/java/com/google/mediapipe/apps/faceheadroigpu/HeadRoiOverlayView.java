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

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;
import com.google.mediapipe.formats.proto.ClassificationProto.Classification;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmark;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList;
import com.google.mediapipe.formats.proto.RectProto.NormalizedRect;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Overlay that handles touch ROI and renders landmarks plus blendshapes. */
public class HeadRoiOverlayView extends View {
  public interface RoiChangeListener {
    void onRoiChanged(NormalizedRect roiRect);
  }

  private static final int MAX_BLENDSHAPES = 12;
  private static final int TOP_BLENDSHAPES = 6;

  private final Paint roiPaint = new Paint();
  private final Paint roiFillPaint = new Paint();
  private final Paint landmarkPaint = new Paint();
  private final Paint textPaint = new Paint();
  private final Paint valuePaint = new Paint();
  private final Paint labelPaint = new Paint();
  private final Paint panelPaint = new Paint();
  private final Paint panelStrokePaint = new Paint();
  private final Paint barBgPaint = new Paint();
  private final Paint barFillPaint = new Paint();

  private final RectF activeRoi = new RectF();
  private final RectF dragRect = new RectF();

  private boolean dragging = false;
  private boolean roiInitialized = false;
  private float dragStartX;
  private float dragStartY;
  private NormalizedLandmarkList landmarks;
  private List<Classification> displayedBlendshapes = Collections.emptyList();
  private double inferenceFps;
  private RoiChangeListener roiChangeListener;

  public HeadRoiOverlayView(Context context) {
    this(context, null);
  }

  public HeadRoiOverlayView(Context context, AttributeSet attrs) {
    this(context, attrs, 0);
  }

  public HeadRoiOverlayView(Context context, AttributeSet attrs, int defStyleAttr) {
    super(context, attrs, defStyleAttr);
    setWillNotDraw(false);

    roiPaint.setColor(Color.rgb(80, 255, 120));
    roiPaint.setStyle(Paint.Style.STROKE);
    roiPaint.setStrokeWidth(4f);
    roiPaint.setAntiAlias(true);

    roiFillPaint.setColor(Color.argb(40, 80, 255, 120));
    roiFillPaint.setStyle(Paint.Style.FILL);

    landmarkPaint.setColor(Color.rgb(255, 215, 0));
    landmarkPaint.setStyle(Paint.Style.FILL);
    landmarkPaint.setAntiAlias(true);

    textPaint.setColor(Color.WHITE);
    textPaint.setTextSize(34f);
    textPaint.setAntiAlias(true);

    valuePaint.setColor(Color.rgb(220, 220, 220));
    valuePaint.setTextSize(28f);
    valuePaint.setAntiAlias(true);
    valuePaint.setTextAlign(Paint.Align.RIGHT);

    labelPaint.set(valuePaint);
    labelPaint.setTextAlign(Paint.Align.LEFT);

    panelPaint.setColor(Color.argb(180, 24, 24, 24));
    panelPaint.setStyle(Paint.Style.FILL);

    panelStrokePaint.setColor(Color.argb(220, 70, 70, 70));
    panelStrokePaint.setStyle(Paint.Style.STROKE);
    panelStrokePaint.setStrokeWidth(2f);

    barBgPaint.setColor(Color.rgb(56, 56, 56));
    barBgPaint.setStyle(Paint.Style.FILL);

    barFillPaint.setColor(Color.rgb(0, 180, 255));
    barFillPaint.setStyle(Paint.Style.FILL);
  }

  public void setRoiChangeListener(RoiChangeListener listener) {
    roiChangeListener = listener;
    notifyRoiChanged();
  }

  public NormalizedRect getCurrentRoi() {
    return toNormalizedRect(activeRoi);
  }

  public void setLandmarks(NormalizedLandmarkList landmarkList) {
    landmarks = landmarkList;
    postInvalidateOnAnimation();
  }

  public void setBlendshapes(List<Classification> classifications) {
    displayedBlendshapes = selectBlendshapes(classifications);
    postInvalidateOnAnimation();
  }

  public void setInferenceFps(double fps) {
    inferenceFps = fps;
    postInvalidateOnAnimation();
  }

  @Override
  protected void onSizeChanged(int w, int h, int oldw, int oldh) {
    super.onSizeChanged(w, h, oldw, oldh);
    if (w <= 0 || h <= 0) {
      return;
    }
    if (!roiInitialized) {
      activeRoi.set(w * 0.2f, h * 0.15f, w * 0.8f, h * 0.85f);
      roiInitialized = true;
      notifyRoiChanged();
    }
  }

  @Override
  public boolean onTouchEvent(MotionEvent event) {
    final float x = clamp(event.getX(), 0f, getWidth());
    final float y = clamp(event.getY(), 0f, getHeight());
    switch (event.getActionMasked()) {
      case MotionEvent.ACTION_DOWN:
        dragging = true;
        dragStartX = x;
        dragStartY = y;
        updateDragRect(x, y);
        postInvalidateOnAnimation();
        return true;
      case MotionEvent.ACTION_MOVE:
        if (!dragging) {
          return false;
        }
        updateDragRect(x, y);
        postInvalidateOnAnimation();
        return true;
      case MotionEvent.ACTION_UP:
      case MotionEvent.ACTION_CANCEL:
        if (!dragging) {
          return false;
        }
        dragging = false;
        updateDragRect(x, y);
        if (dragRect.width() >= 12f && dragRect.height() >= 12f) {
          activeRoi.set(dragRect);
          notifyRoiChanged();
        }
        postInvalidateOnAnimation();
        return true;
      default:
        return super.onTouchEvent(event);
    }
  }

  @Override
  protected void onDraw(Canvas canvas) {
    super.onDraw(canvas);
    if (dragging) {
      canvas.drawRect(dragRect, roiFillPaint);
      canvas.drawRect(dragRect, roiPaint);
    } else {
      canvas.drawRect(activeRoi, roiFillPaint);
      canvas.drawRect(activeRoi, roiPaint);
    }

    canvas.drawText("Head ROI", activeRoi.left + 10f, Math.max(36f, activeRoi.top - 12f), textPaint);
    canvas.drawText(String.format("Infer FPS: %.1f", inferenceFps), 24f, 42f, textPaint);

    if (landmarks != null) {
      final float width = getWidth();
      final float height = getHeight();
      for (NormalizedLandmark landmark : landmarks.getLandmarkList()) {
        canvas.drawCircle(landmark.getX() * width, landmark.getY() * height, 4f, landmarkPaint);
      }
    }

    drawBlendshapePanel(canvas);
  }

  private void drawBlendshapePanel(Canvas canvas) {
    final float panelWidth = Math.min(640f, getWidth() * 0.48f);
    final float left = getWidth() - panelWidth - 20f;
    final float top = 20f;
    final float rowHeight = 34f;
    final float panelHeight = Math.max(150f, 50f + displayedBlendshapes.size() * rowHeight);
    final RectF panel = new RectF(left, top, left + panelWidth, top + panelHeight);
    canvas.drawRoundRect(panel, 16f, 16f, panelPaint);
    canvas.drawRoundRect(panel, 16f, 16f, panelStrokePaint);
    canvas.drawText("Blendshapes", left + 18f, top + 34f, textPaint);

    final float nameLeft = left + 18f;
    final float barLeft = left + panelWidth * 0.48f;
    final float barWidth = panelWidth * 0.30f;
    final float valueRight = left + panelWidth - 18f;

    for (int i = 0; i < displayedBlendshapes.size(); ++i) {
      Classification classification = displayedBlendshapes.get(i);
      float rowTop = top + 48f + i * rowHeight;
      float barTop = rowTop + 6f;
      String label = classification.hasLabel() ? classification.getLabel() : "?";
      canvas.drawText(label, nameLeft, rowTop + 22f, labelPaint);
      RectF barRect = new RectF(barLeft, barTop, barLeft + barWidth, barTop + 18f);
      canvas.drawRoundRect(barRect, 6f, 6f, barBgPaint);
      RectF fillRect =
          new RectF(barLeft, barTop, barLeft + barWidth * clamp(classification.getScore(), 0f, 1f), barTop + 18f);
      canvas.drawRoundRect(fillRect, 6f, 6f, barFillPaint);
      canvas.drawText(String.format("%.2f", classification.getScore()), valueRight, rowTop + 22f, valuePaint);
    }
  }

  private void updateDragRect(float x, float y) {
    dragRect.left = Math.min(dragStartX, x);
    dragRect.top = Math.min(dragStartY, y);
    dragRect.right = Math.max(dragStartX, x);
    dragRect.bottom = Math.max(dragStartY, y);
  }

  private void notifyRoiChanged() {
    if (roiChangeListener != null && getWidth() > 0 && getHeight() > 0) {
      roiChangeListener.onRoiChanged(getCurrentRoi());
    }
  }

  private NormalizedRect toNormalizedRect(RectF rectPx) {
    if (getWidth() <= 0 || getHeight() <= 0) {
      return NormalizedRect.newBuilder()
          .setXCenter(0.5f)
          .setYCenter(0.5f)
          .setWidth(1.0f)
          .setHeight(1.0f)
          .setRotation(0.0f)
          .build();
    }
    float width = clamp(rectPx.width() / getWidth(), 0.01f, 1.0f);
    float height = clamp(rectPx.height() / getHeight(), 0.01f, 1.0f);
    float xCenter = clamp((rectPx.left + rectPx.right) * 0.5f / getWidth(), 0f, 1f);
    float yCenter = clamp((rectPx.top + rectPx.bottom) * 0.5f / getHeight(), 0f, 1f);
    return NormalizedRect.newBuilder()
        .setXCenter(xCenter)
        .setYCenter(yCenter)
        .setWidth(width)
        .setHeight(height)
        .setRotation(0.0f)
        .build();
  }

  private static List<Classification> selectBlendshapes(List<Classification> source) {
    if (source == null || source.isEmpty()) {
      return Collections.emptyList();
    }
    List<Classification> sorted = new ArrayList<>(source);
    Collections.sort(sorted, Comparator.comparing(Classification::getScore).reversed());
    Map<String, Classification> selected = new LinkedHashMap<>();
    for (Classification classification : sorted) {
      String label = classification.hasLabel() ? classification.getLabel() : "?";
      if (selected.size() >= TOP_BLENDSHAPES) {
        break;
      }
      selected.put(label, classification);
    }
    for (Classification classification : sorted) {
      String label = classification.hasLabel() ? classification.getLabel() : "?";
      if (!label.startsWith("mouth")) {
        continue;
      }
      selected.put(label, classification);
      if (selected.size() >= MAX_BLENDSHAPES) {
        break;
      }
    }
    return new ArrayList<>(selected.values());
  }

  private static float clamp(float value, float min, float max) {
    return Math.max(min, Math.min(max, value));
  }
}
