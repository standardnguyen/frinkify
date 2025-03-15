import os
import re
import subprocess
from pathlib import Path

# Import configuration from showenv
try:
    from showenv import source_directory, destination_directory
    # Import frame_rate from showenv, with a fallback default value
    try:
        from showenv import frame_rate
        print(f"Using frame rate from showenv: {frame_rate} fps")
    except ImportError:
        frame_rate = 23.976
        print(f"Frame rate not found in showenv, using default: {frame_rate} fps")
except ImportError:
    print("Error: showenv.py not found. Please create it with source_directory and destination_directory variables.")
    exit(1)

# Check for NVIDIA GPU
has_nvidia_gpu = False
try:
    result = subprocess.run(
        ["nvidia-smi"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    has_nvidia_gpu = result.returncode == 0
except:
    pass

def add_video_filters(mkv_file):
    """
    Analyze the video to determine if it needs deinterlacing or other filters.

    Parameters:
    mkv_file (str): Path to the MKV file

    Returns:
    str: Filter string to use with ffmpeg
    """
    try:
        # Run ffprobe to get video info
        command = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=field_order",
            "-of", "default=noprint_wrappers=1:nokey=1",
            mkv_file
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        field_order = result.stdout.strip()

        # Check if the video is interlaced
        if field_order and field_order not in ["progressive", "unknown"]:
            print(f"Interlaced video detected: {field_order}")
            # Add yadif deinterlacing filter
            return f"fps={frame_rate},yadif=0:-1:0"

        return f"fps={frame_rate}"

    except Exception as e:
        print(f"Error analyzing video: {e}")
        # Default to just setting the frame rate
        return f"fps={frame_rate}"

def parse_episode_info(filename):
    """
    Parse season and episode information from filename.

    Parameters:
    filename (str): The filename to parse

    Returns:
    tuple: (season_number, episode_number) or None if parsing fails
    """
    # Extract season and episode using regex
    match = re.search(r'S(\d+)E(\d+)', filename)
    if match:
        season_num = int(match.group(1))
        episode_num = int(match.group(2))
        return season_num, episode_num
    return None

def find_mkv_for_episode(source_dir, season_num, episode_num):
    """
    Find the corresponding MKV file for a specific episode.

    Parameters:
    source_dir (str): Source directory containing season folders
    season_num (int): Season number
    episode_num (int): Episode number

    Returns:
    str: Path to the MKV file or None if not found
    """
    # Construct the season folder path
    season_folder = os.path.join(source_dir, f"Season {season_num}")

    # Check if the season folder exists
    if not os.path.exists(season_folder):
        print(f"Warning: Season folder not found: {season_folder}")
        return None

    # Search for the episode file
    pattern = f"S{season_num:02d}E{episode_num:02d}"

    for file in os.listdir(season_folder):
        if file.endswith(".mkv") and pattern in file:
            return os.path.join(season_folder, file)

    print(f"Warning: Could not find MKV file for S{season_num:02d}E{episode_num:02d}")
    return None

def extract_all_frames(mkv_file, frames_dir, frame_rate):
    """
    Extract ALL frames from an MKV file at the highest possible resolution using ffmpeg.

    Parameters:
    mkv_file (str): Path to the MKV file
    frames_dir (str): Directory to save extracted frames
    frame_rate (float): Frames per second of the video

    Returns:
    bool: True if successful, False otherwise
    """
    # Create frames directory if it doesn't exist
    os.makedirs(frames_dir, exist_ok=True)

    try:
        # Get video filter string based on analysis
        video_filter = add_video_filters(mkv_file)

        if has_nvidia_gpu:
            # NVIDIA GPU is available, use hardware acceleration
            print("NVIDIA GPU detected - using hardware acceleration")
            command = [
                "ffmpeg",
                "-hwaccel", "cuda",  # Use CUDA hardware acceleration
                "-i", mkv_file,
                "-qscale:v", "1",  # Highest quality setting
                "-vf", video_filter,  # Apply appropriate filters
                "-vsync", "0",  # Each frame is output as soon as it's read
                os.path.join(frames_dir, "frame_%010d.png")  # 10-digit zero padding
            ]
        else:
            # No NVIDIA GPU, fall back to standard processing
            print("No NVIDIA GPU detected - using standard processing")
            command = [
                "ffmpeg",
                "-i", mkv_file,
                "-qscale:v", "1",  # Highest quality setting
                "-vf", f"{video_filter},scale=iw:ih",  # Apply filters and keep original resolution
                "-vsync", "0",  # Each frame is output as soon as it's read
                "-threads", str(os.cpu_count()),  # Use all available CPU cores
                os.path.join(frames_dir, "frame_%010d.png")  # 10-digit zero padding
            ]

        print(f"Executing: {' '.join(command)}")

        # Run ffmpeg command
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Check if the command was successful
        if result.returncode != 0:
            print(f"Error extracting frames: {result.stderr}")
            return False

        # Count the number of extracted frames
        frames_count = len([f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".png")])
        print(f"Successfully extracted {frames_count} frames at maximum quality")
        return True

    except Exception as e:
        print(f"Error during frame extraction: {e}")
        return False

def process_episodes(source_dir, destination_dir, frame_rate):
    """
    Process all episodes in the destination directory to extract all frames from source MKVs.

    Parameters:
    source_dir (str): Source directory containing season folders with MKV files
    destination_dir (str): Destination directory containing processed episode folders
    frame_rate (float): Frames per second of the video

    Returns:
    dict: Statistics about processed episodes
    """
    stats = {
        "episodes_processed": 0,
        "episodes_skipped": 0,
    }

    # Walk through the destination directory
    for season_dir in os.listdir(destination_dir):
        season_path = os.path.join(destination_dir, season_dir)

        # Skip if not a directory or not a season directory
        if not os.path.isdir(season_path) or not season_dir.startswith("Season"):
            continue

        print(f"Processing {season_dir}...")

        # Process each episode directory
        for episode_dir in os.listdir(season_path):
            episode_path = os.path.join(season_path, episode_dir)

            # Skip if not a directory or not an episode directory (SxxExx format)
            if not os.path.isdir(episode_path) or not re.match(r'S\d+E\d+', episode_dir):
                continue

            # Parse episode info
            episode_info = parse_episode_info(episode_dir)
            if not episode_info:
                print(f"Skipping {episode_dir}: Could not parse episode information")
                stats["episodes_skipped"] += 1
                continue

            season_num, episode_num = episode_info

            # Find the corresponding MKV file
            mkv_file = find_mkv_for_episode(source_dir, season_num, episode_num)
            if not mkv_file:
                print(f"Skipping {episode_dir}: No matching MKV file found")
                stats["episodes_skipped"] += 1
                continue

            # Create frames directory
            frames_dir = os.path.join(episode_path, "frames")

            # Check if frames already exist
            if os.path.exists(frames_dir) and len(os.listdir(frames_dir)) > 0:
                # Ask for confirmation before overwriting
                print(f"Frames already exist for {episode_dir}. Skip? (y/n)")
                response = input().strip().lower()
                if response == 'y':
                    print(f"Skipping {episode_dir}")
                    stats["episodes_skipped"] += 1
                    continue

            # Extract all frames
            print(f"Extracting ALL frames for {episode_dir} at maximum quality using frame rate {frame_rate}...")
            success = extract_all_frames(mkv_file, frames_dir, frame_rate)

            if success:
                stats["episodes_processed"] += 1
            else:
                stats["episodes_skipped"] += 1

    return stats

def optimize_wsl_performance():
    """
    Apply WSL-specific optimizations to improve performance.
    """
    # Check if running in WSL
    is_wsl = False
    try:
        with open('/proc/version', 'r') as f:
            if 'microsoft' in f.read().lower():
                is_wsl = True
    except:
        pass

    if is_wsl:
        print("WSL environment detected - applying performance optimizations")

        # Recommend mounting the drive with better performance options
        print("Performance tip: Consider mounting your Windows drives with these options:")
        print("  sudo mount -t drvfs C: /mnt/c -o metadata,case=off,uid=1000,gid=1000,umask=22,fmask=111")

        # Check if destination is on Windows filesystem (slow)
        if any(path in destination_directory for path in ['/mnt/c', '/mnt/d', '/mnt/e']):
            print("Warning: Your destination directory is on a Windows filesystem mount.")
            print("  This will be significantly slower than using the Linux filesystem.")
            print("  Consider using a path in the Linux filesystem (e.g., /home/username/data)")

        # Try to set process priority, but don't require it
        try:
            # For non-sudo users, try to increase priority slightly
            os.nice(5)  # Less aggressive priority increase that may work without sudo
            print("Process priority adjusted for better performance")
        except:
            print("Note: Running with standard process priority - for optimal speed, consider running with sudo")

        return True
    return False

if __name__ == "__main__":
    print(f"Starting FULL frame extraction from {source_directory} to {destination_directory}")
    print("WARNING: This will extract EVERY frame at maximum quality and may require significant disk space!")

    # Apply WSL optimizations
    is_wsl = optimize_wsl_performance()

    print("Continue? (y/n)")

    response = input().strip().lower()
    if response != 'y':
        print("Frame extraction cancelled.")
        exit()

    # Process all episodes
    stats = process_episodes(source_directory, destination_directory, frame_rate)

    print("\nFrame Extraction Complete!")
    print(f"Episodes processed: {stats['episodes_processed']}")
    print(f"Episodes skipped: {stats['episodes_skipped']}")
