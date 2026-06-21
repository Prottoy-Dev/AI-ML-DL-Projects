# People Flow Counting and Heatmap Generation using YOLOv8

This project performs real-time **people detection**, **tracking**, **IN/OUT counting**, and **movement heatmap visualization** using YOLOv8 and tracking algorithms in Python with OpenCV.

---

## 📌 Features

- Person detection using **YOLOv8**.
- Person tracking with unique IDs using **ByteTrack**.
- IN/OUT counting based on movement direction and **line crossing logic**.
- Live frame display with:
  - Total people in the frame
  - IN and OUT counters
  - Bounding box color coding:
    - **White** – default
    - **Green** – counted as IN
    - **Red** – counted as OUT
- Heatmap generation based on person movement:
  - Image output showing accumulated presence
  - Video output showing trails of movement

---

## 🔍 Detection Method

The system uses **Ultralytics YOLOv8** for detecting people (class 0). It uses the `track()` method, which internally applies **ByteTrack** to associate each detection with a unique ID across frames. This helps maintain consistency in counting and heatmap generation.

---

## 📏 Line Coordinates

Two horizontal lines are defined to detect entry and exit movements. These are based on polygon zones and averaged vertically for simplicity:

- **Top Line:** `(15, 367)` to `(1919, 367)`
- **Bottom Line:** `(15, 623)` to `(1910, 605)`

Vertical Y-axis thresholds used:
- `top_y = 367`
- `bottom_y = 614` (average of 623 and 605)

---

## 🔁 IN/OUT Counting Logic

For each tracked person:

- We record the **y-coordinate** of their center point across frames.
- Based on the direction of movement and line crossings, we determine:

#### ✅ IN Condition:
- The person's previous Y-center is **above** `top_y`, and the current Y-center is **below** `top_y`.
- The person is marked as IN, and the bounding box color becomes **green**.

#### ❌ OUT Condition:
- The person's previous Y-center is **below** `bottom_y`, and the current Y-center is **above** `bottom_y`.
- The person is marked as OUT, and the bounding box color becomes **red**.

#### 🟦 Default:
- Before crossing any line, all bounding boxes appear **white**.

This logic prevents duplicate counts and ensures each person is only counted once for IN and OUT events.

---

## 🔥 Heatmap Generation

We visualize where people move over time by plotting the center of their bounding boxes on a black canvas:

- Each frame accumulates the position of tracked people.
- A **Gaussian blur** is applied to simulate movement density.
- The blurred canvas is color-mapped using `cv2.COLORMAP_JET`.
- A heatmap video is produced by overlaying this color map over the original video frames.

### Outputs:
- `heatmap_output.avi`: shows the trails of people over time.
- `heatmap_image.jpg`: final accumulated heatmap image.

