import argparse
# Keep detect_lane_lines() and region_of_interest() unchanged

def process_image(input_path, output_path="lane_detected.jpg"):
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Error: Image not found at {input_path}")
    result = detect_lane_lines(image)
    cv2.imwrite(output_path, result)
    print(f"Saved as: {output_path}")

def process_video(input_path, output_path="lane_detected.mp4"):
    video = cv2.VideoCapture(input_path)
    if not video.isOpened():
        raise FileNotFoundError(f"Error: Video not found at {input_path}")

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
        output_video.write(detect_lane_lines(frame))

    video.release()
    output_video.release()
    print(f"Saved as: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road Lane Line Detection")
    parser.add_argument("--mode", choices=["image", "video"], required=True)
    parser.add_argument("--input", required=True, help="Path to input file")
    args = parser.parse_args()

    if args.mode == "image":
        process_image(args.input)
    else:
        process_video(args.input)
      
