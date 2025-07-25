import sys
import os
import subprocess
import argparse
from pathlib import Path
from typing import Union, List
import json
import re
from PIL import Image
import numpy as np

def parse_partition_string(s):
    """Parse partition string like 'c1b3t9' to get condition, buffer, target counts."""
    pattern = r"c(\d+)b(\d+)t(\d+)"
    m = re.match(pattern, s)
    if not m:
        raise ValueError(f"Invalid partition string: {s}")
    return {
        'condition': int(m.group(1)),
        'buffer': int(m.group(2)),
        'target': int(m.group(3)),
    }

def calc_frames(partition):
    """Calculate frames for each partition using the same logic as calc_num_frames.py."""
    cond_latents = partition['condition']
    buffer_latents = partition['buffer']
    target_latents = partition['target']
    # condition: first latent is 1 frame, rest are 4 frames each
    if cond_latents == 0:
        cond_frames = 0
    elif cond_latents == 1:
        cond_frames = 1
    else:
        cond_frames = 1 + (cond_latents - 1) * 4
    buffer_frames = buffer_latents * 4
    target_frames = target_latents * 4
    total_frames = cond_frames + buffer_frames + target_frames
    return cond_latents, cond_frames, buffer_latents, buffer_frames, target_latents, target_frames, total_frames

def get_video_frame_count(video_path):
    """Get video frame count using ffprobe (from process_FIFO.py)."""
    cmd = [
        'ffprobe', 
        '-v', 'error',
        '-select_streams', 'v:0',
        '-count_packets',
        '-show_entries', 'stream=nb_read_packets',
        '-of', 'json',
        video_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return int(data['streams'][0]['nb_read_packets'])

def get_video_info(video_path: str) -> dict:
    """Get video information using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', '-show_streams', video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json
        info = json.loads(result.stdout)
        
        # Find video stream
        video_stream = None
        for stream in info['streams']:
            if stream['codec_type'] == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            raise ValueError("No video stream found")
        
        return {
            'duration': float(info['format']['duration']),
            'fps': eval(video_stream['r_frame_rate']),  # e.g., "30/1" -> 30.0
            'width': int(video_stream['width']),
            'height': int(video_stream['height']),
            'total_frames': get_video_frame_count(video_path)  # Use the more accurate method
        }
    except subprocess.CalledProcessError as e:
        print(f"Error getting video info: {e}")
        return None

def extract_frames_from_video(video_path: str, num_frames: int, output_dir: str):
    """Extract specified number of frames from video using ffmpeg (from process_FIFO.py)."""
    # Get video frame count
    total_video_frames = get_video_frame_count(video_path)
    if total_video_frames < num_frames:
        print(f"Warning: Video {video_path} has {total_video_frames} frames, but {num_frames} requested")
        num_frames = total_video_frames
    
    # Calculate frame interval for uniform sampling
    frame_interval = round(total_video_frames / num_frames)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract frames using ffmpeg (from process_FIFO.py)
    video_extract_cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'select=not(mod(n\\,{frame_interval})),scale=480:480:force_original_aspect_ratio=decrease,pad=480:480:(ow-iw)/2:(oh-ih)/2,setpts=N/TB',
        '-vframes', str(num_frames),
        '-vsync', '0',
        '-y',
        f"{output_dir}/frame_%04d.jpg"
    ]
    
    try:
        print(f"Extracting {num_frames} frames from {video_path}")
        result = subprocess.run(video_extract_cmd, capture_output=True, text=True, check=True)
        
        # Get the actual extracted frames
        frame_files = sorted(Path(output_dir).glob("frame_*.jpg"))
        return frame_files[:num_frames]
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames from {video_path}: {e}")
        print(f"ffmpeg stderr: {e.stderr}")
        return None

def create_combined_video(condition_images: List[str], target_frames: List[str], 
                         output_path: str, fps: float = 10.0):
    """Create a combined video from condition images and target frames (from process_FIFO.py)."""
    
    # Load and resize all images/frames to 480x480
    all_frames = []
    
    # Process condition images
    for img_path in condition_images:
        img = Image.open(img_path)
        img = img.resize((480, 480), Image.LANCZOS)
        all_frames.append(img)
    
    # Process target frames
    for frame_path in target_frames:
        img = Image.open(frame_path)
        all_frames.append(img)
    
    if not all_frames:
        print("No frames to process")
        return
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create video using ffmpeg (from process_FIFO.py)
    temp_dir = Path("temp_frames")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Save all frames as numbered images
        for i, frame in enumerate(all_frames):
            frame_path = temp_dir / f"frame_{i:04d}.jpg"
            frame.save(frame_path, "JPEG", quality=95)
        
        # Create video from frames (from process_FIFO.py)
        concat_cmd = [
            'ffmpeg',
            '-framerate', str(fps),
            '-i', str(temp_dir / 'frame_%04d.jpg'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'medium',
            '-crf', '23',
            '-y',
            output_path
        ]
        
        print(f"Creating combined video: {output_path}")
        result = subprocess.run(concat_cmd, capture_output=True, text=True, check=True)
        print(f"Successfully created: {output_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
        print(f"ffmpeg stderr: {e.stderr}")
    finally:
        # Clean up temporary files
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def process_video_with_condition(image_folder: str, video_path: str, 
                               mode: str, output_base_dir: str):
    """Process a single video with condition images based on the mode."""
    
    # Parse mode and calculate frames
    partition = parse_partition_string(mode)
    cond_latents, cond_frames, buffer_latents, buffer_frames, target_latents, target_frames, total_frames = calc_frames(partition)
    
    print(f"Mode: {mode}")
    print(f"Condition: {cond_latents} latents ({cond_frames} frames)")
    print(f"Buffer: {buffer_latents} latents ({buffer_frames} frames)")
    print(f"Target: {target_latents} latents ({target_frames} frames)")
    print(f"Total expected frames: {total_frames}")
    print(f"Expected breakdown: {cond_frames} (condition) + {buffer_frames} (buffer) + {target_frames} (target) = {total_frames}")
    
    # Get condition image (i.png)
    image_folder_path = Path(image_folder)
    video_name = Path(video_path).stem
    condition_image_path = image_folder_path / f"{video_name}.png"
    
    if not condition_image_path.exists():
        print(f"Condition image not found: {condition_image_path}")
        return
    
    # Create condition images by duplicating i.png
    condition_images = [str(condition_image_path)] * (cond_frames + buffer_frames)
    print(f"Using {len(condition_images)} copies of {video_name}.png for condition + buffer frames")
    
    # Extract target frames from video
    temp_frames_dir = Path(f"temp_target_frames_{video_name}")
    target_frame_files = extract_frames_from_video(video_path, target_frames, str(temp_frames_dir))
    
    if not target_frame_files:
        print(f"Failed to extract frames from {video_path}")
        return
    
    # Create output path
    output_dir = Path(output_base_dir) / f"_processed_{mode}"
    output_path = output_dir / f"{video_name}.mp4"
    
    # Create combined video
    create_combined_video(condition_images, [str(f) for f in target_frame_files], str(output_path))
    
    # Verify frame count
    print(f"\nVerifying frame count for {video_name}.mp4...")
    verify_video_frames(str(output_path), total_frames)
    

    
    # Clean up temporary files
    import shutil
    if temp_frames_dir.exists():
        shutil.rmtree(temp_frames_dir)

def main():
    parser = argparse.ArgumentParser(description="Create condition videos by combining images and videos")
    parser.add_argument("base_folder", help="Path to base folder containing images/ and videos/ subfolders")
    parser.add_argument("mode", help="Partition mode (e.g., '1c3b9t')")
    
    args = parser.parse_args()
    
    # Check if ffmpeg is available
    if not check_ffmpeg():
        print("Error: ffmpeg is not installed or not in PATH")
        print("Please install ffmpeg first: https://ffmpeg.org/download.html")
        return
    
    # Validate base folder and find subfolders
    base_folder = Path(args.base_folder)
    
    if not base_folder.exists() or not base_folder.is_dir():
        print(f"Base folder not found: {base_folder}")
        return
    
    # Look for images/ and videos/ subfolders
    image_folder = base_folder / "images"
    video_folder = base_folder / "videos"
    
    if not image_folder.exists() or not image_folder.is_dir():
        print(f"Images subfolder not found: {image_folder}")
        print("Please ensure there is an 'images' folder in the base directory")
        return
    
    if not video_folder.exists() or not video_folder.is_dir():
        print(f"Videos subfolder not found: {video_folder}")
        print("Please ensure there is a 'videos' folder in the base directory")
        return
    
    # Get all video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    video_files = sorted([f for f in video_folder.iterdir() 
                         if f.is_file() and f.suffix.lower() in video_extensions])
    
    if not video_files:
        print(f"No video files found in {video_folder}")
        return
    
    # Process each video
    output_base_dir = base_folder
    num_videos = len(video_files)
    
    print(f"Processing {num_videos} videos...")
    
    for i in range(num_videos):
        video_path = video_files[i]
        
        print(f"\nProcessing video {i+1}/{num_videos}: {video_path.name}")
        
        process_video_with_condition(
            str(image_folder), 
            str(video_path), 
            args.mode, 
            str(output_base_dir)
        )
    
    print(f"\nProcessing complete! Output videos saved in: {output_base_dir}/_processed_{args.mode}/")

def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def verify_video_frames(video_path: str, expected_frames: int):
    """Verify that the output video has the expected number of frames."""
    actual_frames = get_video_frame_count(video_path)
    print(f"Actual frames in output video: {actual_frames}")
    print(f"Expected frames: {expected_frames}")
    
    if actual_frames == expected_frames:
        print(f"✓ Frame count verification PASSED: {actual_frames} frames")
        return True
    else:
        print(f"✗ Frame count verification FAILED: expected {expected_frames}, got {actual_frames}")
        print(f"ERROR: Frame count mismatch! Exiting...")
        sys.exit(1)

if __name__ == "__main__":
    main()

"""
Example usage:

python create_cond_video.py /path/to/base_folder 1c3b9t

This will:
1. Look for images/ and videos/ subfolders in the base folder
2. Calculate frames needed: 1 condition + 12 buffer + 36 target = 49 total frames
3. Take 13 copies of i.png from images/ folder (1+12 for condition+buffer)
4. Extract 36 frames from each i.mp4 in videos/ folder
5. Combine them into a single video with 49 frames
6. Save in /path/to/base_folder/_processed_1c3b9t/

Folder structure expected:
/path/to/base_folder/
├── images/
│   ├── video1.png
│   ├── video2.png
│   └── ...
└── videos/
    ├── video1.mp4
    ├── video2.mp4
    └── ...

Requirements:
- ffmpeg must be installed and available in PATH
- PIL (Pillow) for image processing
- Install ffmpeg: https://ffmpeg.org/download.html
- Install Pillow: pip install Pillow
"""
