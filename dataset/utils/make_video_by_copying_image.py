import os
import argparse
import subprocess

def create_video_from_image(image_path, output_path=None):
    """
    Create a video by duplicating an image for 49 frames.
    
    Args:
        image_path (str): Path to the input image
        output_path (str, optional): Path for the output video. If None, will use same name as input with .mp4 extension
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if output_path is None:
        output_path = os.path.splitext(image_path)[0] + '.mp4'
    
    # Create ffmpeg command to resize image and create video
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-loop', '1',  # Loop the input image
        '-i', image_path,  # Input image
        '-vf', 'scale=480:480',  # Resize to 480x480
        '-t', '1.96',  # Duration (49 frames at 25fps = 1.96 seconds)
        '-r', '25',  # Frame rate
        '-c:v', 'libx264',  # Video codec
        '-pix_fmt', 'yuv420p',  # Pixel format for better compatibility
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Video created successfully at: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e.stderr.decode()}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Create a video from an image by duplicating it for 49 frames')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    
    args = parser.parse_args()
    create_video_from_image(args.image_path)

if __name__ == '__main__':
    main()
