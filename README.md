# ⚽ Player Re-Identification Using YOLOv11 & DeepSORT

## 🔍 Project Description

This project presents a complete pipeline for detecting and tracking football players in a single-camera video feed using **YOLOv11** for real-time object detection and **DeepSORT** for reliable multi-object tracking. The system assigns each player a unique and **persistent identity (ID)**, allowing consistent re-identification across frames—even if a player temporarily exits the scene or is occluded.

The implementation is optimized for performance and modularity, making it a strong foundation for downstream tasks such as **pass detection**, **heatmap generation**, or **statistical player analysis** in football analytics.

---

## 🎯 Objective

The main goals of this project were:

* To detect football players using a fine-tuned YOLOv11 model.
* To assign and maintain consistent tracking IDs using DeepSORT.
* To visually annotate the video with bounding boxes, frame numbers, and player IDs.
* To output a new video showing consistent re-identification across the clip.

---

## 🛠 Technologies and Tools

| Tool / Library            | Purpose                                      |
| ------------------------- | -------------------------------------------- |
| **YOLOv11 (Ultralytics)** | Real-time object detection (players only)    |
| **DeepSORT**              | Multi-object tracking & ID assignment        |
| **OpenCV**                | Video frame reading, writing, and annotation |
| **NumPy**                 | Array operations & vector math               |

---

## 📂 Project Directory Structure

```
project-folder/
│
├── best.pt                      # Custom YOLOv11 weights for player detection
├── 15sec_input_720p.mp4         # Input match footage (15-second clip)
├── output_reid_v2.mp4           # Output video with annotated player IDs
├── pass_tracking.py             # Python script containing full implementation
├── README.md                    # This file
```

---

## ⚙️ How It Works

### 1. Load Model and Input Video

```python
model = YOLO("best.pt")
cap = cv2.VideoCapture("15sec_input_720p.mp4")
```

A fine-tuned YOLOv11 model is loaded to detect players in the video. Only detections with class `0` (Player) are used.

---

### 2. Initialize Tracker

```python
tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.4)
```

DeepSORT is used to assign and maintain player identities across frames based on spatial and appearance features.

---

### 3. Frame-by-Frame Processing

Each frame is processed using the following steps:

* Object detection with YOLOv11
* Filter detections to include only players
* Pass bounding boxes to DeepSORT for tracking
* Draw bounding boxes, IDs, and frame numbers on the video

```python
detections.append(([x1, y1, x2, y2], conf, "player"))
```

```python
cv2.putText(frame, f'ID: {track_id}', ...)
```

---

### 4. Save the Output

The processed frames are saved to a video file using OpenCV:

```python
out = cv2.VideoWriter('output_reid_v2.mp4', fourcc, fps, (width, height))
```

Once all frames are processed, the video writer and video stream are properly released.

---

## 🧪 Output Description

* **output\_reid\_v2.mp4**: This is the final annotated video.

  * Every player has a bounding box drawn around them.
  * Player IDs remain consistent even if players disappear and re-enter the frame.
  * Each frame is labeled for easier debugging and reference.

---

## 📸 Sample Frame (Optional)

> You can insert snapshots or example video frames here using:
>
> ```
> ![Sample Output](images/frame_123.png)
> ```

---

## 🧩 Features

* Fine-tuned YOLOv11-based detection (optimized for football players).
* High-performance real-time tracking using DeepSORT.
* Robust ID assignment and re-identification even during occlusions.
* Easy-to-extend pipeline (e.g. for event detection, player stats, tactical analysis).

---

## 🚀 Future Improvements

* ⚽ **Pass Detection** – Track the ball and identify pass sequences using proximity logic.
* 📊 **Heatmap Visualization** – Show player movement zones and coverage.
* ⏱️ **Player Stats** – Add distance run, time on screen, or possession percentage.
* 📈 **Live Analytics Integration** – Stream processed data to dashboards or web apps.

---

## 🧠 Learning Outcomes

Through this project, I deepened my understanding of:

* Building real-time object detection pipelines.
* Re-identifying multiple objects across frames under complex conditions.
* Integrating custom YOLOv11 models with tracking logic.
* Video annotation and export with OpenCV.

---

## ▶️ Running the Project

1. Install dependencies:

```bash
pip install ultralytics opencv-python deep_sort_realtime
```

2. Run the script:

```bash
python pass_tracking.py
```

3. Check the generated file:

```
output_reid_v2.mp4
```

---

## 🔐 License

This project is for educational and research use. Please reference appropriately if used in derivative work.


