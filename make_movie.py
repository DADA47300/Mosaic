import cv2
import numpy as np
import sys

def linear_range(val1: int, val2: int, n: int):
    return np.linspace(val1, val2, n).astype(int).tolist()

def rect_morphing(rect_src: tuple, rect_dest: tuple, nb_steps: int):
    x1_values = linear_range(rect_src[0], rect_dest[0], nb_steps)
    y1_values = linear_range(rect_src[1], rect_dest[1], nb_steps)
    x2_values = linear_range(rect_src[2], rect_dest[2], nb_steps)
    y2_values = linear_range(rect_src[3], rect_dest[3], nb_steps)
    
    rects = [(x1, y1, x2, y2) for x1, y1, x2, y2 in zip(x1_values, y1_values, x2_values, y2_values)]
    return rects

def makemovie(img: np.ndarray, rect_src: tuple, rect_dest: tuple, output_filename: str, w: int, h: int, fps: int, duration: int):
    total_frames = fps * duration
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))

    intermediate_rects = rect_morphing(rect_src, rect_dest, total_frames)
    
    for rect in intermediate_rects:
        x1, y1, x2, y2 = rect
        roi = img[y1:y2, x1:x2]
        resized_roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)
        video.write(resized_roi)

    video.release()
    print(f"Video saved as {output_filename}")
    

def compute_scale(img, next_img, index):
    ratio_width = img.shape[1] / next_img.shape[1]
    ratio_height = img.shape[0] / next_img.shape[0]
    max_ratio = max(ratio_width, ratio_height)
    
    if index >= 6:
        default_scale = 1.75
    else:
        default_scale = 2
    
    if max_ratio < 1.7:
        max_scale = default_scale
    elif max_ratio > 2:
        max_scale = 2.35
    else:
        max_scale = default_scale

    print("max_scale calculated:", max_scale)
    return max_scale

def resize_and_write_frames(img, max_scale, output_video_path, duration, fps, target_size):
    img = cv2.resize(img, target_size)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, target_size)
    center = (target_size[0] // 2, target_size[1] // 2)
    
    base_scale = np.linspace(1, max_scale, int(duration * fps))
    for scale in base_scale:
        M = cv2.getRotationMatrix2D(center, 0, scale)
        frame = cv2.warpAffine(img, M, target_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        video.write(frame)
    video.release()

def read_and_append_video(current_video_path, video_writer):
    cap = cv2.VideoCapture(current_video_path)
    if not cap.isOpened():
        print(f"Failed to open {current_video_path}")
        return False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_writer.write(frame)
    cap.release()
    return True

def zoom_x2_mp4(image_path, output_video_path, next_image_path, duration, index, fps=24, target_size=(2048, 2048), is_last=False):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot find the file: {image_path}")

    max_scale = 5 if is_last else compute_scale(img, cv2.imread(next_image_path), index)
    resize_and_write_frames(img, max_scale, output_video_path, duration, fps, target_size)

def recursive_zoom_mp4(input_prefix, i_max, final_video_path, duration, fps=24, video_writer=None, target_size=(2048, 2048), i=1):
    if i > i_max:
        if video_writer:
            print(f"{final_video_path} completed")
            video_writer.release()
        return

    image_path = f"{input_prefix}_{i}.jpg"
    next_image_path = f"{input_prefix}_{i+1}.jpg" if i < i_max else None
    current_video_path = f"{input_prefix}_{i}.mp4"
    
    zoom_x2_mp4(image_path, current_video_path, next_image_path, duration, i, fps, target_size, is_last=(i == i_max))
    
    if video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(final_video_path, fourcc, fps, target_size)

    if not read_and_append_video(current_video_path, video_writer):
        return

    recursive_zoom_mp4(input_prefix, i_max, final_video_path, duration, fps, video_writer, target_size, i + 1)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: make_movie.py <MASTER_IMG> <PREFIX_IMG> <DURATION> <ZOOM_MAGNITUDE> <OUTPUT_FILENAME>")
        sys.exit(1)

    master_img = sys.argv[1]
    input_prefix = sys.argv[2]
    duration = int(sys.argv[3])
    zoom_magnitude = int(sys.argv[4])
    output_filename = sys.argv[5]

    i_max = zoom_magnitude
    recursive_zoom_mp4(input_prefix, i_max, output_filename, duration)