# Live Video Vehicle Damage Inspection System Plan - Hybrid Approach (YOLOv11 & Gemini 2.0 Flash)

## 1. Overview

This document outlines the plan for developing a live video vehicle damage inspection system using a hybrid approach. The system will use YOLOv11 for real-time damage detection and Gemini 2.0 Flash for enhanced report generation. Gemini 2.0 Flash will receive a representative video frame and accumulated YOLOv11 model outputs from a 5-second interval to generate context-aware and periodically updated reports.

## 2. System Components

- **Video Capture:** Captures video frames from a live camera feed (e.g., webcam, IP camera).
- **Real-time Object Detection (YOLOv11):** Employs the trained YOLOv11 model for real-time damage detection in video frames.
- **Bounding Box Overlay:** Overlays bounding boxes and labels on the video frames to visually highlight detected damages based on YOLOv11 output.
- **Live Report Generation (Gemini 2.0 Flash Enhanced - Periodic Updates):** Leverages Gemini 2.0 Flash to generate a live, updating text report every 5 seconds. Gemini 2.0 Flash receives a representative video frame and accumulated damage detections from YOLOv11 (sampled at 2 frames per second) over a 5-second interval as input for enhanced report generation.
- **Display/Output:** Displays the processed video stream with bounding boxes and the live text report.

## 3. Technical Stack

- **Programming Language:** Python
- **CV Library:** OpenCV (cv2) - for video capture and bounding box overlay.
- **Deep Learning Framework:** PyTorch - for YOLOv11 model.
- **Object Detection Model:** YOLOv11n (pre-trained and fine-tuned).
- **GenAI Model:** Google Gemini 2.0 Flash - for enhanced report generation.
- **GenAI Framework:** LangChain - for managing interaction with Gemini API and prompts.

## 4. Workflow Steps (Real-time Processing Loop) - Hybrid with Periodic Reporting

1.  **Initialize:**

    - Load YOLOv11 model.
    - Initialize video capture from camera (OpenCV).
    - Create display windows (OpenCV).
    - Initialize empty report data structure.
    - Initialize frame counter and time tracker for periodic reporting.
    - Initialize a list to accumulate YOLO outputs for the 5-second interval.

2.  **Main Loop (per frame):**
    1.  **Read Frame:** Capture a frame from the live video feed (OpenCV).
    2.  **Frame Sampling (for YOLO):** Process YOLOv11 inference on approximately every 15th frame (to achieve ~2 FPS processing, assuming 30 FPS camera). Increment frame counter.
    3.  **Conditional YOLOv11 Inference:** If the frame counter modulo 15 is 0:
        - **Preprocess Frame (for YOLO):** Preprocess the frame for YOLOv11 input.
        - **Run YOLOv11 Inference:** Pass the preprocessed frame through the YOLOv11 model to get damage detections (bounding boxes, classes, scores).
        - **Process Detections:** Filter detections based on confidence threshold.
        - **Accumulate YOLO Outputs:** Add the processed detections to the accumulation list for the current 5-second interval.
    4.  **Draw Bounding Boxes:** Draw bounding boxes and labels on the **current frame** using OpenCV, based on the **latest YOLOv11 detections** (even if YOLO is not run on every frame, display the most recent detections).
    5.  **Display Video:** Display the frame with bounding boxes in a video window (OpenCV).
    6.  **Time Check for Report Update:** Check if 5 seconds have elapsed since the last report update.
    7.  **Conditional Report Generation (every 5 seconds):** If 5 seconds have elapsed:
        - **Select Representative Frame:** Choose a representative frame from the last 5 seconds (e.g., the latest processed frame).
        - **Send to Gemini 2.0 Flash for Report Generation:**
          - Send the **representative video frame** and the **accumulated YOLOv11 detections** (from the last 5 seconds) to the Gemini 2.0 Flash API.
          - Prompt Gemini 2.0 Flash to generate a descriptive report, utilizing both visual information from the frame and the accumulated structured detection data from YOLOv11.
        - **Receive and Display Live Report:** Receive the GenAI-enhanced text report from Gemini 2.0 Flash and display the updated report.
        - **Clear Accumulation List:** Clear the YOLO output accumulation list to start accumulating for the next 5-second interval.
        - **Reset Time Tracker:** Reset the time tracker for the next 5-second interval.
    8.  **Display Live Report:** Display the GenAI-enhanced text report (the report will update approximately every 5 seconds).
    9.  **Loop:** Repeat steps 4.1-4.9 for the next frame.
    10. **Exit Condition:** Implement a way to exit the loop.

## 5. GenAI Integration Plan

### 5.1 Phase 2.1: Enhanced Reporting (Periodic Updates)

- **Goal:** Generate periodically updated (every 5 seconds) and descriptive damage reports using Gemini 2.0 Flash.
- **Input to GenAI:** **Representative video frame (from the last 5 seconds) and accumulated YOLO output (damage types, bounding boxes, confidence scores) from the past 5 seconds (sampled at ~2 FPS).**
- **Prompt Engineering:** Design prompts for **Gemini 2.0 Flash** to generate detailed reports summarizing detected damages over the last 5 seconds in natural language, utilizing both the visual information from the representative video frame and the accumulated structured data from YOLO output. The prompt should instruct Gemini 2.0 Flash to generate a report that reflects the damages detected within the 5-second interval.
- **Output from GenAI:** Descriptive text report, updated every 5 seconds.
- **Integration:** Display the GenAI-generated report in the live output, updating it approximately every 5 seconds, using **Gemini 2.0 Flash** for report generation with multimodal input and periodic updates.

### 5.2 Phase 2.2: Severity Estimation (Future)

- **Goal:** Use GenAI to estimate the severity of detected damages.
- **Input to GenAI (Severity Prompt):** YOLO output (damage types, confidence scores).
- **Severity Prompt:** Prompt Claude-3 Opus to classify damage severity (e.g., minor, moderate, severe).
- **Severity Output:** Severity assessment for each damage type.
- **Report Integration:** Integrate severity estimations into the GenAI-enhanced report.

### 5.3 Phase 2.3: Repair Recommendations (Future)

- **Goal:** Leverage GenAI to suggest preliminary repair recommendations.
- **Knowledge Base:** Potentially integrate an external knowledge source about vehicle damage and repairs.
- **Repair Recommendation Prompt:** Prompt Claude-3 Opus to suggest repair actions based on damage types and severity.
- **Recommendation Output:** Repair recommendations.
- **Report Integration:** Include repair recommendations in the GenAI-enhanced report.

### 5.4 Phase 2.4: Contextual Understanding (Future)

- **Goal:** Enhance report relevance by incorporating contextual information (vehicle type, inspection scenario).
- **Context Input:** Capture or provide contextual information to GenAI.
- **Contextual Prompt:** Modify prompts to consider context for report generation.
- **Contextualized Output:** More relevant and informative reports.

## 6. Next Steps (Implementation)

1.  **Research Gemini 2.0 Flash API (Multimodal Input & Performance):** Investigate API documentation focusing on handling both images and structured data for report generation tasks and API performance characteristics for periodic calls.
2.  **Set up Python environment:** Install required libraries (OpenCV, PyTorch, Ultralytics YOLO, LangChain, Google Gemini API client).
3.  **API Key for Gemini 2.0 Flash:** Obtain a Google Gemini API key.
4.  **Write Python script for real-time video processing (Hybrid with Periodic Reporting):** Implement video capture, YOLOv11 inference (sampled), Gemini 2.0 Flash API calls (sending representative frame and accumulated YOLOv11 detections every 5 seconds), bounding box drawing, and periodically updated enhanced report display.
5.  **Test with webcam and Gemini 2.0 Flash API (Hybrid Periodic Reporting):** Test the script and evaluate real-time performance, report quality, API integration, and report update frequency.
6.  **Iterate and improve (Hybrid Periodic Reporting):** Optimize performance, refine prompts for multimodal input to Gemini 2.0 Flash, enhance report generation, adjust sampling and reporting intervals as needed, and address any API limitations.
