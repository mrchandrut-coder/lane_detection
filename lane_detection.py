import argparse
import cv2
import numpy as np


# ---------------------------------------
# Region of Interest
# ---------------------------------------
def region_of_interest(edges):
    height, width = edges.shape[:2]

    mask = np.zeros_like(edges)

    polygon = np.array([[
        (0, height),
        (width, height),
        (int(width * 0.60), int(height * 0.58)),
        (int(width * 0.40), int(height * 0.58))
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)

    return cv2.bitwise_and(edges, mask)


# ---------------------------------------
# Lane Detection
# ---------------------------------------
def detect_lane_lines(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blur, 50, 150)

    # Select road area
    roi = region_of_interest(edges)

    # Detect lines
    lines = cv2.HoughLinesP(
        roi,
        1,
        np.pi / 180,
        threshold=40,
        minLineLength=40,
        maxLineGap=120
    )

    output = image.copy()

    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Ignore horizontal lines
            if abs(slope) < 0.4:
                continue

            height = image.shape[0]

            y_bottom = height
            y_top = int(height * 0.58)

            x_bottom = int(x1 + (y_bottom - y1) / slope)
            x_top = int(x1 + (y_top - y1) / slope)

            lane = (x_bottom, y_bottom, x_top, y_top)

            # Left lane
            if slope < 0:
                left_lines.append(lane)
            # Right lane
            else:
                right_lines.append(lane)

    # Draw left lane
    if left_lines:
        left = np.median(left_lines, axis=0).astype(int)
        cv2.line(
            output,
            tuple(left[:2]),
            tuple(left[2:]),
            (0, 255, 0),
            7
        )

    # Draw right lane
    if right_lines:
        right = np.median(right_lines, axis=0).astype(int)
        cv2.line(
            output,
            tuple(right[:2]),
            tuple(right[2:]),
            (0, 255, 0),
            7
        )

    # Combine original + detected lanes
    result = cv2.addWeighted(
        image,
        0.75,
        output,
        0.25,
        0
    )

    return result


# ---------------------------------------
# Image Processing
# ---------------------------------------
def process_image(input_path, output_path="lane_detected.jpg"):
    image = cv2.imread(input_path)

    if image is None:
        raise FileNotFoundError(f"Error: Image not found at '{input_path}'")

    result = detect_lane_lines(image)
    cv2.imwrite(output_path, result)

    print("Image processing completed!")
    print(f"Saved as: {output_path}")


# ---------------------------------------
# Video Processing
# ---------------------------------------
def process_video(input_path, output_path="lane_detected.mp4"):
    video = cv2.VideoCapture(input_path)

    if not video.isOpened():
        raise FileNotFoundError(f"Error: Video not found at '{input_path}'")

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = video.read()
        if not ret:
            break

        result = detect_lane_lines(frame)
        output_video.write(result)

    video.release()
    output_video.release()

    print("Video processing completed!")
    print(f"Saved as: {output_path}")


# ---------------------------------------
# Main Entry Point
# ---------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road Lane Line Detection")
    parser.add_argument("--mode", choices=["image", "video"], required=True, help="Processing mode")
    parser.add_argument("--input", required=True, help="Path to input image or video")
    parser.add_argument("--output", default=None, help="Optional custom output path")

    args = parser.parse_args()

    if args.mode == "image":
        out = args.output if args.output else "lane_detected.jpg"
        process_image(args.input, out)
    elif args.mode == "video":
        out = args.output if args.output else "lane_detected.mp4"
        process_video(args.input, out)
