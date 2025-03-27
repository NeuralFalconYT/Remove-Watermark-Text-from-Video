gpu=True 
import easyocr
reader = easyocr.Reader(['ch_sim','en'],gpu=gpu) # this needs to run only once to load the model into memory

import cv2
import os
import cv2
import numpy as np
import shutil
from concurrent.futures import ThreadPoolExecutor
import re
import subprocess

def extract_frames(video_path, output_folder):
    if os.path.exists(output_folder):
      shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop when video ends
        
        frame_path = os.path.join(output_folder, f"{frame_count:06d}.png")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")







# Initialize text reader

def remove_watermark(image, blur_type="strong_gaussian"):
    results = reader.readtext(image)  # Detect text regions
    
    for (bbox, text, prob) in results:
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        x1, y1 = top_left
        x2, y2 = bottom_right
        roi = image[y1:y2, x1:x2]

        if blur_type == "strong_gaussian":
            blurred_roi = cv2.GaussianBlur(roi, (25, 25), 50)
        elif blur_type == "pixelation":
            h, w = roi.shape[:2]
            temp = cv2.resize(roi, (8, 8), interpolation=cv2.INTER_LINEAR)
            blurred_roi = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        elif blur_type == "median":
            blurred_roi = cv2.medianBlur(roi, 21)
        elif blur_type == "motion":
            size = 25
            kernel = np.zeros((size, size))
            kernel[:, size//2] = 1
            kernel = kernel / kernel.sum()
            blurred_roi = cv2.filter2D(roi, -1, kernel)
        elif blur_type == "bilateral":
            blurred_roi = cv2.bilateralFilter(roi, d=15, sigmaColor=75, sigmaSpace=75)
        elif blur_type == "box":
            blurred_roi = cv2.blur(roi, (25, 25))
        elif blur_type == "stacked":
            temp = cv2.GaussianBlur(roi, (15, 15), 25)
            blurred_roi = cv2.medianBlur(temp, 15)
        elif blur_type == "adaptive":
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
            blurred = cv2.GaussianBlur(roi, (25, 25), 25)
            blurred_roi = np.where(mask[..., None] > 0, blurred, roi)
        else:
            blurred_roi = cv2.GaussianBlur(roi, (25, 25), 50)

        image[y1:y2, x1:x2] = blurred_roi
    return image
    # return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)




def process_frame(frame_path, save_path, blur_type):
    image = cv2.imread(frame_path)
    
    if image is None:
        print(f"Failed to load: {frame_path}")  # Debugging step
        return
    
    no_watermark_image = remove_watermark(image, blur_type=blur_type)
    
    output_file = os.path.join(save_path, os.path.basename(frame_path))
    success = cv2.imwrite(output_file, no_watermark_image)
    
    if not success:
        print(f"Failed to save: {output_file}")  # Debugging step

def batch_process(blur_type="median", batch_size=100):
    input_folder = "./frames"
    output_folder = "./clean"
    
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    frame_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith((".jpg", ".png"))]
    
    with ThreadPoolExecutor() as executor:
        executor.map(process_frame, frame_paths, [output_folder] * len(frame_paths), [blur_type] * len(frame_paths))

    print(f"Processing complete! {len(frame_paths)} frames saved to {output_folder}")




def get_video_fps(video_path):
    """Extract FPS from the original video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def sorted_files(directory):
    """Returns a list of sorted .png files based on numeric order."""
    files = [f for f in os.listdir(directory) if f.endswith(".png")]
    files.sort(key=lambda f: int(re.search(r'\d+', f).group()) if re.search(r'\d+', f) else float('inf'))
    return [os.path.join(directory, f) for f in files]

def create_video_chunks(frame_dir, output_dir, fps, batch_size=100):
    """Creates chunked videos from frames in batches."""
    
    # Remove old "chunks" folder if exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    sorted_images = sorted_files(frame_dir)
    
    total_chunks = (len(sorted_images) // batch_size) + (1 if len(sorted_images) % batch_size else 0)
    
    for i in range(total_chunks):
        chunk_frames = sorted_images[i * batch_size:(i + 1) * batch_size]
        if not chunk_frames:
            continue
        
        chunk_folder = os.path.join(output_dir, f"chunk_{i+1}")
        os.makedirs(chunk_folder, exist_ok=True)
        
        # Copy frames to a temp folder
        for j, frame in enumerate(chunk_frames):
            frame_dest = os.path.join(chunk_folder, f"{j:05d}.png")  # Zero-padded filenames
            shutil.copy(frame, frame_dest)

        # Generate video from frames
        chunk_output = os.path.join(output_dir, f"{i+1}.mp4")
        ffmpeg_cmd = f'ffmpeg -y -framerate {fps} -i "{chunk_folder}/%05d.png" -c:v libx264 -pix_fmt yuv420p "{chunk_output}"'
        subprocess.run(ffmpeg_cmd, shell=True, check=True)

        # Cleanup temp chunk folder
        shutil.rmtree(chunk_folder)

    print(f"✅ All {total_chunks} video chunks created in {output_dir}")

def vido_chunks(video_path):
  # Extract original FPS
  fps = get_video_fps(video_path)
  if fps is None:
      raise ValueError("Failed to retrieve FPS from video.")

  # Define folders
  frame_dir = "./clean"
  output_dir = "./chunks"
  # Process frames into video chunks
  create_video_chunks(frame_dir, output_dir, fps, batch_size=100)

import os
import re
import uuid

def sanitize_file(file_path):
    folder = os.path.dirname(file_path)  
    text, ext = os.path.splitext(os.path.basename(file_path))  
    
    # Keep alphabets, spaces, and underscores only
    text = re.sub(r'[^a-zA-Z_ ]', '', text)  
    text = text.lower().strip()             
    text = text.replace(" ", "_")           
    
    # Truncate or handle empty text
    truncated_text = text[:20] if len(text) > 20 else text if len(text) > 0 else "empty"
    
    # Generate a random string for uniqueness
    random_string = uuid.uuid4().hex[:8].upper()
    
    # Construct the new file name
    # file_name = f"{folder}/{truncated_text}_{random_string}{ext}"
    file_name = f"{truncated_text}_{random_string}{ext}"
    return file_name
def upload_file(video_path):
  if os.path.exists("./upload"):
    shutil.rmtree("./upload")   
  os.makedirs("./upload",exist_ok=True)
  new_path=sanitize_file(video_path)
  new_path=f"./upload/{new_path}"
  shutil.copy(video_path,new_path)
  return new_path



import os
import re
import subprocess
def sorted_video_files(directory):
    """Returns a list of full paths of .mp4 files sorted by the numeric part of the filename."""
    files = [f for f in os.listdir(directory) if f.endswith(".mp4")]

    # Extract the numeric part using regex and sort
    files.sort(key=lambda f: int(re.search(r'\d+', f).group()) if re.search(r'\d+', f) else float('inf'))

    # Convert filenames to full paths
    full_paths = [os.path.join(directory, f) for f in files]

    return full_paths

def marge_video(gpu=True):
    os.makedirs("./result/",exist_ok=True)
    output_path=f"./result/no_water_mark.mp4"
    video_list=sorted_video_files("./chunks")
    with open("./join.txt", "w") as f:
        for video in video_list:
            f.write(f"file '{video}'\n")
    if gpu:
      join_command = f'ffmpeg -hwaccel cuda -f concat -safe 0 -i ./join.txt -c copy "{output_path}" -y'
    else:
      join_command = f'ffmpeg -f concat -safe 0 -i ./join.txt -c copy "{output_path}" -y'
    subprocess.run(join_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path
def recover_audio(upload_path):
  output_path=f"./result/no_water_mark.mp4"
  audio_path="./upload/temp.wav"
  os.makedirs("./result/",exist_ok=True)
  base_name=os.path.basename(upload_path)
  save_path=f"./result/{base_name.replace('.mp4','_no_watermark.mp4')}"
#   save_path=upload_path.replace(".mp4","_no_watermark.mp4")
  var=os.system(f"ffmpeg -i {upload_path} -q:a 0 -map a {audio_path} -y")
  if var==0:
    var2=os.system(f"ffmpeg -i {output_path} -i {audio_path} -c:v copy -map 0:v:0 -map 1:a:0 -shortest {save_path} -y")
    if var2==0:
      return save_path
  return None 
def video_watermark_remover(video_path, blur_type="median"):
  global gpu
  upload_path=upload_file(video_path)
  extract_frames(upload_path, "./frames")
  batch_process(blur_type=blur_type)
  vido_chunks(upload_path)
  marge_video(gpu=gpu)
  save_path=recover_audio(upload_path) 
  return save_path


import gradio as gr 
import click

def gradio_interface(video_file, blur_type):
    vid_path=video_watermark_remover(video_file, blur_type=blur_type)
    return vid_path,vid_path

blur_types = ["strong_gaussian", "median"]  

demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.File(label="Upload Video"),
        gr.Dropdown(choices=blur_types, label="Blur Type", value="median")  # Default to median
    ],
    outputs=[gr.File(label="Download Video"),gr.Video(label="Play Video")],
    title="Video Watermark Remover",
    description="Upload a video, and this tool will remove watermarks using blurring techniques."
)

# demo.launch()

@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def main(debug, share):
    demo.queue().launch(debug=debug, share=share)
if __name__ == "__main__":
    main()
