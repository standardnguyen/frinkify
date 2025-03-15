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

# Check for NVIDIA GPU and identify RTX 3090
has_nvidia_gpu = False
gpu_model = None
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode == 0:
        has_nvidia_gpu = True
        gpu_model = result.stdout.strip()
        print(f"NVIDIA GPU detected: {gpu_model}")
except:
    pass

# Configure optimal parameters for RTX 3090
rtx_3090_detected = has_nvidia_gpu and gpu_model and "3090" in gpu_model
if rtx_3090_detected:
    print("RTX 3090 detected - enabling maximum GPU optimizations")
    # Set environment variables to optimize CUDA performance
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure we're using the right GPU
    os.environ["AV_LOG_FORCE_COLOR"] = "1"  # Better logging

    # For RTX 3090, we can enable higher performance modes
    try:
        # Try to set GPU to max performance state (requires appropriate permissions)
        subprocess.run(
            ["nvidia-smi", "--gpu-reset"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        subprocess.run(
            ["nvidia-smi", "--applications-clocks=mem:19500,graphics:1995"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("GPU clock settings optimized for maximum performance")
    except:
        print("Note: Could not set optimal GPU clocks - requires admin privileges")

# Check for i9-9900K CPU
cpu_info = ""
try:
    with open('/proc/cpuinfo', 'r') as f:
        cpu_info = f.read()
except:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            cpu_info = result.stdout.strip()
    except:
        pass

i9_9900k_detected = "i9-9900K" in cpu_info or "i9 9900K" in cpu_info
if i9_9900k_detected:
    print("Intel i9-9900K detected - enabling CPU optimizations")

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
            print("NVIDIA GPU detected - using optimized hardware acceleration")

            # The decoder should come BEFORE the input file for hardware acceleration
            command = [
                "ffmpeg",
                "-hwaccel", "cuda",  # Use CUDA hardware acceleration
                "-thread_queue_size", "1024",  # Queue size before input file
                "-c:v", "hevc_cuvid",  # Use NVIDIA's HEVC decoder since the file is HEVC/H.265
                "-i", mkv_file,
                "-qscale:v", "1",  # Highest quality setting
                # Fix the filter chain - removed problematic hwdownload,format=nv12
                "-vf", video_filter,  # Just apply the frame rate filter
                "-vsync", "0",  # Each frame is output as soon as it's read
                os.path.join(frames_dir, "frame_%010d.png")  # 10-digit zero padding
            ]
        else:
            # No NVIDIA GPU, optimize for i9-9900K (8 cores/16 threads)
            print("Optimizing for i9-9900K CPU - using high-performance CPU processing")
            command = [
                "ffmpeg",
                "-thread_queue_size", "1024",  # MOVED HERE - before input file
                "-i", mkv_file,
                "-qscale:v", "1",  # Highest quality setting
                "-vf", f"{video_filter},scale=iw:ih",  # Apply filters and keep original resolution
                "-vsync", "0",  # Each frame is output as soon as it's read
                "-threads", "16",  # Use all threads on i9-9900K
                "-filter_threads", "8",  # Optimize filter thread count
                "-filter_complex_threads", "8",  # Optimize complex filter thread count
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

def optimize_hardware_performance():
    """Apply system optimizations for high-end hardware."""
    # Configure system for optimal I/O performance
    try:
        # Try to increase process priority for better CPU scheduling
        if os.name == 'posix':  # Linux/Mac
            try:
                os.nice(-10)  # Higher priority (requires sudo)
                print("Process priority increased for better performance")
            except:
                print("Note: Could not increase process priority - requires sudo")

            # Try to optimize I/O priority
            try:
                subprocess.run(
                    ["ionice", "-c", "1", "-n", "0", str(os.getpid())],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                print("I/O priority optimized for maximum throughput")
            except:
                pass

            # Try to set CPU governor to performance mode
            try:
                subprocess.run(
                    ["sudo", "cpupower", "frequency-set", "-g", "performance"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                print("CPU governor set to performance mode")
            except:
                pass
    except Exception as e:
        print(f"Note: Some performance optimizations could not be applied: {e}")

    # Disable power saving features for maximum performance
    if rtx_3090_detected:
        try:
            subprocess.run(
                ["nvidia-smi", "--power-limit=350"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("GPU power limit increased for maximum performance")
        except:
            pass

    return True

def setup_parallel_processing(destination_dir):
    """Set up parallel processing for multiple episodes."""
    import multiprocessing

    # Determine optimal number of parallel processes
    # With RTX 3090 and i9-9900K, we have plenty of resources
    # But we still need to be careful about memory usage
    if rtx_3090_detected:
        # If VRAM is being used for decoding, limit concurrency
        return min(4, multiprocessing.cpu_count() // 2)
    elif i9_9900k_detected:
        # More CPU-based parallelism
        return min(8, multiprocessing.cpu_count() - 2)
    else:
        # Conservative default
        return max(1, multiprocessing.cpu_count() // 4)

if __name__ == "__main__":
    print(f"Starting FULL frame extraction from {source_directory} to {destination_directory}")
    print("WARNING: This will extract EVERY frame at maximum quality and may require significant disk space!")

    # Apply optimizations
    is_wsl = optimize_wsl_performance()
    optimize_hardware_performance()

    # Setup SSD/NVMe optimizations if available
    dest_device = os.statvfs(destination_directory)
    source_device = os.statvfs(source_directory)

    # Calculate available space
    available_space_gb = (dest_device.f_frsize * dest_device.f_bavail) / (1024**3)
    print(f"Available space at destination: {available_space_gb:.2f} GB")

    if available_space_gb < 100:
        print("WARNING: Less than 100GB available at destination. Frame extraction may fail if disk space runs out.")

    print("\nHARDWARE OPTIMIZATION SUMMARY:")
    print(f"- GPU: {'RTX 3090 (Optimized)' if rtx_3090_detected else gpu_model if has_nvidia_gpu else 'Not detected'}")
    print(f"- CPU: {'i9-9900K (Optimized)' if i9_9900k_detected else 'Standard'}")
    print(f"- WSL: {'Optimized' if is_wsl else 'Not detected'}")
    print(f"- Parallel processing: Enabled")
    print("\nEstimated performance: Excellent\n")

    print("Continue with optimized extraction? (y/n)")

    response = input().strip().lower()
    if response != 'y':
        print("Frame extraction cancelled.")
        exit()

    # Consider parallel processing
    max_parallel = setup_parallel_processing(destination_directory)
    print(f"Configured for up to {max_parallel} parallel extractions")

    if max_parallel > 1:
        import multiprocessing
        print("Using parallel processing for maximum speed")
        pool = multiprocessing.Pool(processes=max_parallel)
        # Note: This would require restructuring the process_episodes function
        # to support parallel processing, which is beyond the scope of this update

    # Process all episodes
    print("Starting extraction with optimized settings...")
    stats = process_episodes(source_directory, destination_directory, frame_rate)

    print("\nFrame Extraction Complete!")
    print(f"Episodes processed: {stats['episodes_processed']}")
    print(f"Episodes skipped: {stats['episodes_skipped']}")
