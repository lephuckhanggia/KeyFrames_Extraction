import cv2
import os
from itertools import product
from PIL import Image, ImageChops
from timeit import default_timer as timer
import csv

# Define paths
videos_folder = r'C:\AI Chalenge 2024\Data 2024\Round_1\Video_Full'
keyframes_root_folder = r'D:\Gia_Projects\github.com\lephuckhanggia\KeyFrames_Extract\KeyFrames'
csv_folder = r'D:\Gia_Projects\github.com\lephuckhanggia\KeyFrames_Extract\CSV'

# Define your similarity threshold and frame skip rate
difference_threshold = 0.1  # Difference less than this value means frames are considered "similar"
skip_rate = 10  # Adjust this to skip every Nth frame


def summarise(img: Image.Image) -> Image.Image:
    """Summarise an image into a 16 x 16 image"""
    resized = img.resize((16, 16))
    return resized


def difference(img1: Image.Image, img2: Image.Image) -> float:
    """Find the difference between the two images"""
    diff = ImageChops.difference(img1, img2)

    acc = 0
    width, height = diff.size
    for w, h in product(range(width), range(height)):
        r, g, b = diff.getpixel((w, h))
        acc += (r + g + b) / 3

    average_diff = acc / (width * height)
    normalised_diff = average_diff / 255
    return normalised_diff


def process_video(video_path, keyframes_folder, csv_path):
    """Process a single video file"""
    vid = cv2.VideoCapture(video_path)
    i = 0
    frame_num = 1

    # Get the video FPS
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    start = timer()
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Time", "FrameID", "Difference"])  # Write CSV header

        prev_frame = None
        prev_time_sec = None
        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                break
            if i % skip_rate == 0:
                # Get the current time of the frame
                current_time_sec = vid.get(cv2.CAP_PROP_POS_MSEC) / 1000

                # Convert the current frame to PIL format for processing
                curr_frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # If there is a previous frame, calculate the difference
                if prev_frame is not None:
                    img1 = summarise(prev_frame)
                    img2 = summarise(curr_frame_pil)
                    diff = difference(img1, img2)

                    # Only save frames if the difference exceeds the threshold
                    if diff > difference_threshold:
                        print(
                            f"Saving frame {i}: Difference {diff}, Time: {prev_time_sec:.2f}s -> {current_time_sec:.2f}s")
                        frameID = current_time_sec * fps
                        writer.writerow([frame_num, f"{current_time_sec:.2f}", frameID, diff])  # Save to CSV

                        # Save the current frame to the specified folder
                        frame_path = os.path.join(keyframes_folder, f'{frame_num}.jpg')
                        cv2.imwrite(frame_path, frame)
                        frame_num += 1
                    else:
                        print(f"Skipping frame {i} due to similarity (Difference: {diff})")

                # Update the frame
                prev_frame = curr_frame_pil
                prev_time_sec = current_time_sec
            i += 1

    # Release the video capture object
    print(f"Time to process video: {timer() - start} seconds")
    vid.release()


def main():
    # Ensure the keyframes and CSV folders exist
    if not os.path.exists(keyframes_root_folder):
        os.makedirs(keyframes_root_folder)
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)

    # Process each video in the videos folder
    for video_file in os.listdir(videos_folder):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(videos_folder, video_file)
            video_name = os.path.splitext(video_file)[0]

            # Create specific folders for each video
            keyframes_folder = os.path.join(keyframes_root_folder, video_name)
            if not os.path.exists(keyframes_folder):
                os.makedirs(keyframes_folder)

            csv_path = os.path.join(csv_folder, f"{video_name}.csv")

            # Process the video
            process_video(video_path, keyframes_folder, csv_path)


if __name__ == "__main__":
    main()
