import threading
import cv2
import keyboard
import numpy as np
import subprocess
from ultralytics import YOLO
import supervision as sv


# Load YOLOv8 model
model = YOLO('yolov8s.pt')  # Change this to the path of your YOLOv8 model if needed

# Set up the input and output RTSP streams
input_rtsp = 'rtsp://localhost:8554/eufy_stream'
output_rtsp = 'rtsp://localhost:8554/eufy_stream_ai'

# Open the input stream using OpenCV
cap = cv2.VideoCapture(input_rtsp)

# Set up the output stream using FFmpeg
ffmpeg_command = [
    'ffmpeg',
    '-re',  # Read input at native frame rate
    '-f', 'rawvideo',  # Input format
    '-pix_fmt', 'bgr24',  # Pixel format
    '-s', '1920x1080',  # Frame size
    '-r', '15',  # Frame rate
    '-i', '-',  # Input from stdin
    '-c:v', 'libx264',  # Video codec
    '-pix_fmt', 'yuv420p',  # Output pixel format
    '-preset', 'veryfast',  # Encoding speed/quality
    '-f', 'rtsp',  # Output format
    output_rtsp
]

output_pipe = subprocess.Popen(
    ffmpeg_command,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# Function to read and display stderr output
def read_stderr(pipe):
    while True:
        line = pipe.readline()
        if not line:
            break
        print(f"FFmpeg stderr: {line.strip()}")

# Function to read and display stderr output
def read_stdout(pipe):
    while True:
        line = pipe.readline()
        if not line:
            break
        print(f"FFmpeg stdout: {line.strip()}")

# Start a thread to read stderr
stderr_thread = threading.Thread(target=read_stderr, args=(output_pipe.stderr,))
stderr_thread.start()

# Start a thread to read stderr
stdout_thread = threading.Thread(target=read_stdout, args=(output_pipe.stdout,))
stdout_thread.start()

# Function to perform person detection and return annotated frame
def detect_and_annotate(frame):
    results = model(frame)  # Perform detection
    detections = sv.Detections.from_ultralytics(results[0])
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_frame = bounding_box_annotator.annotate(frame, detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections)
    return annotated_frame

# Process the input stream
frame_width = 1920
frame_height = 1080
frame_size = frame_width * frame_height * 3

# Define the codec and create VideoWriter object
output_file = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter(output_file, fourcc, 15.0, (1920, 1080))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection and annotation
    annotated_frame = detect_and_annotate(frame)

    if keyboard.is_pressed('q'):
        print("Exiting loop...")
        break

    out.write(annotated_frame)

    try:
        # Write the annotated frame to the buffer file
        output_pipe.stdin.write(annotated_frame.tobytes())
    except:
        # Rebuild the output if it breaks for any reason
        output_pipe = subprocess.Popen(
           ffmpeg_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

# Clean up
out.release()
cap.release()
output_pipe.stdin.close()
