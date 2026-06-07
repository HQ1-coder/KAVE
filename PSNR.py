import argparse
import math

import cv2
import numpy as np

def calculate_psnr(video_path1, video_path2):
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    if not cap1.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path1}")
    if not cap2.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path2}")
    
    psnr_total = 0.0
    frame_count = 0
    
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        

        frame1 = cv2.resize(frame1, (frame2.shape[1], frame2.shape[0]))
        

        mse = np.mean((frame1 - frame2) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            max_pixel = 255.0
            psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        
        psnr_total += psnr
        frame_count += 1
    
    cap1.release()
    cap2.release()
    
    if frame_count == 0:
        raise ValueError("No readable frames found in one or both videos.")

    return psnr_total / frame_count


def main():
    parser = argparse.ArgumentParser(description="Calculate average PSNR between two videos.")
    parser.add_argument("cover_video", help="Path to the cover video.")
    parser.add_argument("stego_video", help="Path to the stego video.")
    args = parser.parse_args()

    psnr_value = calculate_psnr(args.cover_video, args.stego_video)
    print(f"Average PSNR: {psnr_value:.2f} dB")


if __name__ == "__main__":
    main()
