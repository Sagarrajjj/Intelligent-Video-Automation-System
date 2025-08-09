#Skills Used:
#Python Programming
#Video Downloading (yt_dlp library)
#Audio Extraction (os module for running ffmpeg)
#Speech-to-Text (speech_recognition library)
#Computer Vision (cv2 library for object detection)
#Deep Learning (Conceptual understanding of YOLO for object detection)
#Natural Language Processing (transformers library for text classification)
#System Command Execution (os module)
import cv2
import speech_recognition as sr
from transformers import pipeline
import os
import yt_dlp


# Step 1: Download YouTube video and extract audio using yt-dlp
def download_youtube_video(youtube_url):
    ydl_opts = {
        'format': 'bestaudio/best',  # Download best quality audio
        'outtmpl': 'youtube_audio.mp4',  # Output filename
        'noplaylist': True,  # Don't download playlist if URL is a playlist
        'quiet': True  # Suppress verbose messages
    }

    # Download the YouTube audio
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    return 'youtube_audio.mp4'


# Step 2: Extract Audio from YouTube Video and Convert to Text (Speech-to-Text)
def speech_to_text(video_file):
    recognizer = sr.Recognizer()
    # Extract audio from video using ffmpeg (if it's in video format)
    audio_file = "audio_from_video.wav"
    os.system(f"ffmpeg -i {video_file} -ab 160k -ac 2 -ar 44100 -vn {audio_file}")

    # Perform speech recognition
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data)


# Step 3: Object Detection in YouTube Video (optional for command enhancement)
def detect_objects(video_file):
    cap = cv2.VideoCapture(video_file)
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare the frame for YOLO object detection
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        output_layers = net.getUnconnectedOutLayersNames()
        detections = net.forward(output_layers)

        # Process detections
        print("Objects detected in frame...")

    cap.release()
    cv2.destroyAllWindows()


# Step 4: Analyze the Speech Command Using NLP
def analyze_command(command_text):
    nlp_model = pipeline("text-classification")
    result = nlp_model(command_text)
    return result


# Step 5: Execute Action Based on Command
def execute_action(command_text):
    if "open email" in command_text.lower():
        os.system("start chrome https://mail.google.com")
    elif "run script" in command_text.lower():
        os.system("python example_script.py")
    elif "show video" in command_text.lower():
        os.system("vlc video_file.mp4")
    else:
        print(f"No action found for: {command_text}")


# Main Function: Process YouTube Video and Perform Tasks
def process_youtube_video(youtube_url):
    print("Downloading YouTube video...")
    video_file = download_youtube_video(youtube_url)

    print("Extracting and analyzing audio...")
    command_text = speech_to_text(video_file)
    print(f"Command recognized: {command_text}")

    # Optional: Analyze the video for object detection
    print("Analyzing video for object detection...")
    detect_objects(video_file)

    print("Analyzing command with NLP...")
    analyzed_command = analyze_command(command_text)
    print(f"Analyzed command: {analyzed_command}")

    print("Executing action...")
    execute_action(command_text)


# Run the Program
if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=WnyNMbx4TEE"  # Replace with your YouTube URL
    process_youtube_video(youtube_url)
