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
import java.util.List;

/** Overlay that handles touch ROI and renders landmarks plus blendshapes. */
public class HeadRoiOverlayView extends View {
  public interface RoiChangeListener {
    void onRoiChanged(NormalizedRect roiRect);
  }

  // Lip landmark indices from MediaPipe 478-point face mesh (outer + inner lips).
  private static final int[] MOUTH_LANDMARK_INDICES = {
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95,
    146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314,
    317, 318, 321, 324, 375, 402, 405, 409, 415
  };

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

    if (landmarks != null && landmarks.getLandmarkCount() > 0) {
      final float width = getWidth();
      final float height = getHeight();
      for (NormalizedLandmark lm : landmarks.getLandmarkList()) {
        canvas.drawCircle(lm.getX() * width, lm.getY() * height, 4f, landmarkPaint);
      }
    }

    drawBlendshapePanel(canvas);
  }

  private void drawBlendshapePanel(Canvas canvas) {
    if (displayedBlendshapes.isEmpty()) {
      return;
    }
    final int count = displayedBlendshapes.size();
    final int colRows = (count + 1) / 2; // rows per column
    final float rowHeight = 36f;
    final float headerH = 42f;
    final float padding = 16f;
    final float panelHeight = headerH + colRows * rowHeight + padding;
    final float panelTop = 0f;
    final float panelWidth = getWidth();

    // Background
    final RectF panel = new RectF(0f, panelTop, panelWidth, panelHeight);
    canvas.drawRect(panel, panelPaint);
    canvas.drawLine(0f, panelHeight, panelWidth, panelHeight, panelStrokePaint);

    canvas.drawText("Mouth blendshapes", padding, panelTop + 30f, textPaint);

    // Two columns
    final float colWidth = panelWidth / 2f;
    final float nameRatio = 0.44f;
    final float barRatio = 0.30f;
    final float valRatio = 0.14f; // right-aligned, remaining space

    for (int i = 0; i < count; i++) {
      final int col = i / colRows;
      final int row = i % colRows;
      final Classification cls = displayedBlendshapes.get(i);
      final String label = cls.hasLabel() ? cls.getLabel() : "?";
      final float score = cls.getScore();

      final float colLeft = col * colWidth;
      final float rowTop = panelTop + headerH + row * rowHeight;
      final float textBaseline = rowTop + rowHeight - 8f;
      final float barTop = rowTop + 8f;
      final float barH = rowHeight - 18f;

      final float nameLeft = colLeft + padding;
      final float barLeft = colLeft + colWidth * nameRatio;
      final float barWidth = colWidth * barRatio;
      final float valueRight = colLeft + colWidth - padding;

      canvas.drawText(label, nameLeft, textBaseline, labelPaint);

      final RectF barRect = new RectF(barLeft, barTop, barLeft + barWidth, barTop + barH);
      canvas.drawRoundRect(barRect, 4f, 4f, barBgPaint);
      final float filled = barWidth * clamp(score, 0f, 1f);
      if (filled > 0f) {
        final RectF fillRect = new RectF(barLeft, barTop, barLeft + filled, barTop + barH);
        canvas.drawRoundRect(fillRect, 4f, 4f, barFillPaint);
      }
      canvas.drawText(String.format("%.2f", score), valueRight, textBaseline, valuePaint);
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
    List<Classification> mouth = new ArrayList<>();
    for (Classification cls : source) {
      if (cls.hasLabel() && cls.getLabel().startsWith("mouth")) {
        mouth.add(cls);
      }
    }
    // Keep original model order (stable, predictable layout).
    return mouth;
  }

  private static float clamp(float value, float min, float max) {
    return Math.max(min, Math.min(max, value));
  }
}
