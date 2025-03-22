# n8n Workflow Plan for Batch Image Inspection and Reporting

## 1. Overview

This document outlines the plan for an n8n workflow to perform batch vehicle damage inspection and generate a consolidated report. The workflow will take a set of input images, process each image for damage detection using a FastAPI backend, and generate a single, consolidated report summarizing the damages found in all images, enhanced by GenAI (Gemini 2.0 Flash).

## 2. n8n Workflow Steps

1.  **HTTP Request Trigger (or Manual Trigger):**

    - **Trigger Type:** HTTP Request trigger (POST endpoint to receive a list of image URLs or image files).
    - **Input Data:** Expect a JSON payload in the HTTP POST request containing a list of image URLs or image files (base64 encoded or multipart form data).

2.  **Split in Batches (Item Lists):**

    - **Node Type:** "Item Lists" node (or "Split Batch" node).
    - **Purpose:** Process each image individually in the workflow by iterating through the list of images provided in the input.

3.  **Image Processing Loop (per image):**

    1.  **Image Retrieval (if URLs provided):**
        - **Node Type:** "HTTP Request" node.
        - **Purpose:** If image URLs are provided, download each image from its URL.
    2.  **Call Damage Detection Model (FastAPI):**
        - **Node Type:** "HTTP Request" node.
        - **Purpose:** Call the FastAPI backend API endpoint for damage detection for the current image. Send the image data to the FastAPI API.
        - **Input:** Image data (either downloaded from URL or directly provided).
        - **Output:** JSON response from FastAPI containing damage detection results for the image.
    3.  **Extract Model Output:**
        - **Node Type:** "Function" node or "JSON Parse" node.
        - **Purpose:** Parse the JSON response from FastAPI to extract damage detection results (bounding boxes, classes, scores) for the current image.
    4.  **Collect Results (Item Lists - Merge):**
        - **Node Type:** "Item Lists" - "Merge" mode (or "Merge" node after "Split Batch").
        - **Purpose:** Aggregate the damage detection results for each image into a single data structure.

4.  **Report Generation (GenAI - Gemini 2.0 Flash):**

    - **Node Type:** "Function" node.
    - **Purpose:** Generate a consolidated report summarizing damages from all processed images using GenAI (Gemini 2.0 Flash).
    - **Input to GenAI:** Aggregated damage detection results from all images. Optionally, representative images can also be sent for visual context.
    - **Prompt Engineering:** Design a prompt for Gemini 2.0 Flash to generate a consolidated report summarizing damages across all input images.
    - **Output from GenAI:** Consolidated text report summarizing damages for all input images.

5.  **Report Delivery (Email, Storage, HTTP Response):**

    - **Node Type:** "Email" node, "Google Drive" node, "AWS S3" node, "Local File System" node, "HTTP Response" node.
    - **Purpose:** Deliver the generated report via email, storage, or HTTP response.
    - **Input:** The GenAI-generated consolidated report text.

6.  **HTTP Response (if HTTP Trigger):**
    - **Node Type:** "HTTP Response" node.
    - **Purpose:** Send a response back to the client who initiated the workflow via HTTP request.

## 3. Technical Stack

- **n8n:** Workflow automation platform.
- **FastAPI Backend:** Existing API for damage detection model.
- **GenAI Model:** Google Gemini 2.0 Flash (for report generation).
- **LangChain (potentially):** For interaction with Gemini API.

## 4. Next Steps (Implementation)

1.  **Set up n8n:** Install and configure n8n.
2.  **Design n8n workflow:** Create the batch image inspection and reporting workflow in n8n.
3.  **Modify FastAPI backend (if needed):** Ensure the FastAPI backend API endpoint is ready for n8n integration.
4.  **Test and Iterate:** Thoroughly test the n8n workflow with a batch of images.
5.  **Deploy n8n workflow:** Deploy the n8n workflow to a production environment.
