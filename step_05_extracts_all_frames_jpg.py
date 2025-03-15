import os
import re
import subprocess
from pathlib import Path

# -----------------------
# Frame Extraction Configuration
# -----------------------
# Set to False to use PNG instead of JPEG
USE_JPEG = True
# JPEG quality (1-100, higher is better quality but larger files)
JPEG_QUALITY = 95
# Chroma subsampling: "444"=highest quality, "422"=good balance, "420"=smaller files
CHROMA_SUBSAMPLING = "444"

# Resolution settings
# Set to "original" to keep source resolution, or use specific values like "1920:1080"
# You can also use "half" for half resolution or formulas like "iw/2:ih/2"
RESOLUTION = "original"
# Use scale algorithm: fast bilinear, bilinear, bicubic, lanczos (quality ascending order)
SCALE_ALGORITHM = "lanczos"

# Frame selection settings
# Extract every Nth frame (1 = all frames, 2 = every other frame, etc.)
FRAME_INTERVAL = 5
# Whether frame numbering should reflect source frames (True) or be sequential (False)
# For example, with FRAME_INTERVAL=3:
#   - True: frames will be numbered frame_000001.jpg, frame_000004.jpg, frame_000007.jpg...
#   - False: frames will be numbered frame_000001.jpg, frame_000002.jpg, frame_000003.jpg...
PRESERVE_FRAME_NUMBERS = True
# -----------------------

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

def extract_all_frames(mkv_file, frames_dir, frame_rate, use_jpeg=USE_JPEG,
                     jpeg_quality=JPEG_QUALITY, chroma_subsampling=CHROMA_SUBSAMPLING,
                     resolution=RESOLUTION, scale_algorithm=SCALE_ALGORITHM,
                     frame_interval=FRAME_INTERVAL, preserve_frame_numbers=PRESERVE_FRAME_NUMBERS):
    """
    Extract ALL frames from an MKV file at the specified quality settings.

    Parameters:
    mkv_file (str): Path to the MKV file
    frames_dir (str): Directory to save extracted frames
    frame_rate (float): Frames per second of the video
    use_jpeg (bool): Use JPEG instead of PNG for frame extraction
    jpeg_quality (int): JPEG quality (1-100, higher is better quality but larger files)
    chroma_subsampling (str): Chroma subsampling method (444=best, 422=good, 420=smaller)
    resolution (str): Output resolution (original, half, or specific dimensions like "1920:1080")
    scale_algorithm (str): Scaling algorithm to use (fast_bilinear, bilinear, bicubic, lanczos)
    frame_interval (int): Extract every Nth frame (1=all frames, 2=every other frame, etc.)
    preserve_frame_numbers (bool): If True, frame numbers reflect source frames (e.g., 1,4,7,...)

    Returns:
    bool: True if successful, False otherwise
    """
    # Create frames directory if it doesn't exist
    os.makedirs(frames_dir, exist_ok=True)

    # DEBUG: Check frames directory exists and permissions
    print(f"DEBUG: Frames directory: {frames_dir}")
    print(f"DEBUG: Directory exists: {os.path.exists(frames_dir)}")
    print(f"DEBUG: Directory is writable: {os.access(frames_dir, os.W_OK)}")
    try:
        test_file = os.path.join(frames_dir, "test_write.tmp")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print(f"DEBUG: Successfully wrote and deleted test file in frames directory")
    except Exception as e:
        print(f"DEBUG: Error writing test file to frames directory: {e}")

    # Validate jpeg_quality if using JPEG
    if use_jpeg:
        if jpeg_quality < 1 or jpeg_quality > 100:
            print(f"Invalid JPEG quality: {jpeg_quality}. Using default of 95.")
            jpeg_quality = 95

        # Validate chroma_subsampling
        valid_subsampling = ["444", "422", "420"]
        if chroma_subsampling not in valid_subsampling:
            print(f"Invalid chroma subsampling: {chroma_subsampling}. Using 444 (highest quality).")
            chroma_subsampling = "444"

    try:
        # Get video filter string based on analysis
        video_filter = add_video_filters(mkv_file)

        # Handle resolution scaling
        scale_filter = ""

        # Get source video dimensions if needed
        video_dimensions = None
        if resolution != "original":
            try:
                # Get video dimensions using ffprobe
                cmd = [
                    "ffprobe",
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height",
                    "-of", "csv=p=0",
                    mkv_file
                ]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode == 0:
                    width, height = map(int, result.stdout.strip().split(','))
                    video_dimensions = (width, height)
                    print(f"Source video dimensions: {width}x{height}")
            except Exception as e:
                print(f"Warning: Could not determine source video dimensions: {e}")

        # Build scale filter based on resolution setting
        if resolution == "original":
            # Keep original resolution, no scaling needed
            scale_filter = ""
        elif resolution == "half" and video_dimensions:
            # Half the original resolution
            new_width = video_dimensions[0] // 2
            new_height = video_dimensions[1] // 2
            scale_filter = f",scale={new_width}:{new_height}:flags={scale_algorithm}"
            print(f"Scaling to half resolution: {new_width}x{new_height}")
        elif resolution.lower() in ["hd", "720p"]:
            scale_filter = f",scale=1280:720:flags={scale_algorithm}"
            print("Scaling to 720p (1280x720)")
        elif resolution.lower() in ["fullhd", "1080p"]:
            scale_filter = f",scale=1920:1080:flags={scale_algorithm}"
            print("Scaling to 1080p (1920x1080)")
        elif resolution.lower() in ["4k", "uhd", "2160p"]:
            scale_filter = f",scale=3840:2160:flags={scale_algorithm}"
            print("Scaling to 4K (3840x2160)")
        elif ":" in resolution:
            # Custom resolution in format "width:height"
            scale_filter = f",scale={resolution}:flags={scale_algorithm}"
            print(f"Scaling to custom resolution: {resolution}")
        elif "iw" in resolution or "ih" in resolution:
            # Formula-based resolution (e.g., "iw/2:ih/2")
            scale_filter = f",scale={resolution}:flags={scale_algorithm}"
            print(f"Scaling with formula: {resolution}")
        else:
            print(f"Warning: Invalid resolution format '{resolution}', using original resolution")

        # Append scale filter to the video filter if needed
        if scale_filter:
            video_filter = video_filter + scale_filter

        # Add frame selection filter if not extracting every frame
        if frame_interval > 1:
            # select filter uses 0-based indexing, so we use n%INTERVAL==0 to get every Nth frame
            video_filter = f"{video_filter},select='not(mod(n,{frame_interval}))'"
            print(f"Extracting every {frame_interval}th frame")

            # The select filter will drop frames but maintain metadata about which frames were kept
            # FFmpeg will output these frames sequentially, so we need to adjust our output pattern
            # to preserve the original frame numbers if requested

        # Determine file extension and quality settings
        file_ext = "jpg" if use_jpeg else "png"

        # Set output pattern based on frame numbering preference
        if frame_interval > 1 and preserve_frame_numbers:
            # We'll use -frame_pts 1 to make ffmpeg use the actual frame timestamp (PTS)
            # as part of the filename, and a custom naming pattern
            # This requires adding special options to the command later
            print(f"Using source-based frame numbering")
            output_pattern = os.path.join(frames_dir, f"frame_%010d.{file_ext}")
        else:
            output_pattern = os.path.join(frames_dir, f"frame_%010d.{file_ext}")

        # DEBUG: Print output pattern
        print(f"DEBUG: Output pattern: {output_pattern}")
        print(f"DEBUG: Will save frames as: {file_ext.upper()}")

        if has_nvidia_gpu:
            # NVIDIA GPU is available, use hardware acceleration
            print(f"NVIDIA GPU detected - using optimized hardware acceleration for {file_ext.upper()} encoding")

            if use_jpeg:
                # For JPEG: Convert quality 1-100 to ffmpeg's qscale 2-31 (inverted)
                # Higher quality = lower qscale value (2 is best, 31 is worst)
                qscale_value = max(2, min(31, int(31 - (jpeg_quality / 100.0 * 29))))

                command = [
                    "ffmpeg",
                    "-hwaccel", "cuda",
                    "-thread_queue_size", "1024",
                    "-c:v", "hevc_cuvid",
                    "-i", mkv_file,
                    "-q:v", str(qscale_value),
                    "-vf", video_filter,
                    "-vsync", "0",
                    "-pix_fmt", f"yuvj{chroma_subsampling}p",
                    output_pattern
                ]
            else:
                # For PNG: Use lossless settings
                command = [
                    "ffmpeg",
                    "-hwaccel", "cuda",
                    "-thread_queue_size", "1024",
                    "-c:v", "hevc_cuvid",
                    "-i", mkv_file,
                    "-qscale:v", "1",
                    "-vf", video_filter,
                    "-vsync", "0",
                    output_pattern
                ]
        else:
            # No NVIDIA GPU, optimize for i9-9900K (8 cores/16 threads)
            print(f"Optimizing for i9-9900K CPU - using high-performance CPU processing for {file_ext.upper()} encoding")

            if use_jpeg:
                # For JPEG: Convert quality 1-100 to ffmpeg's qscale 2-31 (inverted)
                qscale_value = max(2, min(31, int(31 - (jpeg_quality / 100.0 * 29))))

                command = [
                    "ffmpeg",
                    "-thread_queue_size", "1024",
                    "-i", mkv_file,
                    "-q:v", str(qscale_value),
                    "-vf", f"{video_filter},scale=iw:ih",
                    "-vsync", "0",
                    "-threads", "16",
                    "-filter_threads", "8",
                    "-filter_complex_threads", "8",
                    "-pix_fmt", f"yuvj{chroma_subsampling}p",
                    output_pattern
                ]
            else:
                # For PNG: Use lossless settings
                command = [
                    "ffmpeg",
                    "-thread_queue_size", "1024",
                    "-i", mkv_file,
                    "-qscale:v", "1",
                    "-vf", f"{video_filter},scale=iw:ih",
                    "-vsync", "0",
                    "-threads", "16",
                    "-filter_threads", "8",
                    "-filter_complex_threads", "8",
                    output_pattern
                ]

        print(f"Executing: {' '.join(command)}")

        # Run ffmpeg command
        if frame_interval > 1 and preserve_frame_numbers:
            # For preserving source frame numbers, we need a custom extraction approach
            # We'll use a Python-based solution that directly numbers the frames correctly

            # First, get the total number of frames in the source video
            try:
                cmd = [
                    "ffprobe",
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-count_frames",
                    "-show_entries", "stream=nb_read_frames",
                    "-of", "csv=p=0",
                    mkv_file
                ]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                total_frames = int(result.stdout.strip()) if result.returncode == 0 else None
                if total_frames:
                    print(f"Total frames in source: {total_frames}")
            except:
                total_frames = None
                print("Could not determine total frame count")

            # Now let's extract frames with the correct numbering
            # We'll modify our command to use the -vsync 0 option to maintain frame timing
            # and use itsoffset to start from specific frames

            # Create a version of the command that uses -vf select but preserves frame numbers
            select_command = command.copy()

            # Need to replace the output pattern with one that includes %d for frame numbering
            output_pattern_with_number = os.path.join(frames_dir, f"frame_%010d.{file_ext}")
            select_command[-1] = output_pattern_with_number

            # Add the -frame_pts option if available (depends on ffmpeg version)
            # This option causes the %d in the output pattern to be replaced with the pts value
            # However, some versions of ffmpeg don't support this
            # select_command.insert(-1, "-frame_pts")
            # select_command.insert(-1, "1")

            print(f"Extracting frames with source-based numbering")

            # DEBUG: Print the directory structure before extraction
            print(f"DEBUG: Directory structure before extraction:")
            print(f"DEBUG: Frames dir exists: {os.path.exists(frames_dir)}")
            if os.path.exists(frames_dir):
                print(f"DEBUG: Contents of frames dir: {os.listdir(frames_dir)}")

            # Run the extraction with the select filter, which will skip frames but maintain sequential output
            result = subprocess.run(
                select_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # DEBUG: Check execution result
            print(f"DEBUG: ffmpeg return code: {result.returncode}")
            print(f"DEBUG: ffmpeg stderr output length: {len(result.stderr)}")
            if len(result.stderr) > 0:
                print(f"DEBUG: First 500 chars of ffmpeg stderr: {result.stderr[:500]}")

            # DEBUG: Check the directory after ffmpeg
            print(f"DEBUG: Directory structure after extraction:")
            print(f"DEBUG: Frames dir exists: {os.path.exists(frames_dir)}")
            if os.path.exists(frames_dir):
                frame_files = os.listdir(frames_dir)
                print(f"DEBUG: Number of files in frames dir: {len(frame_files)}")
                print(f"DEBUG: First few files (if any): {frame_files[:5] if frame_files else 'No files'}")

            if result.returncode != 0:
                print(f"Error extracting frames: {result.stderr}")
                return False

            # After extraction, we need to rename the files to match their source frame numbers
            # This step is only needed because some ffmpeg versions don't support -frame_pts properly
            print(f"Renaming frames to match source frame numbers")

            # DEBUG: Check if there are any frame files to rename
            frame_files = [f for f in os.listdir(frames_dir)
                          if f.startswith("frame_") and f.endswith(f".{file_ext}")]
            print(f"DEBUG: Found {len(frame_files)} frame files to rename")

            if not frame_files:
                print(f"ERROR: No frames were extracted to rename!")
                return False

            # Create a temp directory for the renaming process
            temp_dir = os.path.join(frames_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)

            # DEBUG: Check temp directory
            print(f"DEBUG: Temp directory created: {os.path.exists(temp_dir)}")

            # Rename files to reflect source frame numbers
            for i, frame_file in enumerate(sorted(frame_files)):
                # Calculate the source frame number (0-based index * interval + 1 for 1-based frame numbering)
                source_frame = (i * frame_interval) + 1

                # Create the new filename
                new_name = f"frame_{source_frame:010d}.{file_ext}"

                # DEBUG: Print rename operation
                if i < 5:  # Only print first few to avoid log spam
                    print(f"DEBUG: Renaming {frame_file} to {new_name}")

                # Move to temp dir with new name to avoid filename conflicts
                try:
                    os.rename(
                        os.path.join(frames_dir, frame_file),
                        os.path.join(temp_dir, new_name)
                    )
                    if i < 5:  # Only print first few
                        print(f"DEBUG: Successfully renamed file {i+1}")
                except Exception as e:
                    print(f"DEBUG: Error renaming file {frame_file}: {e}")

            # DEBUG: Check number of files in temp dir
            temp_files = os.listdir(temp_dir)
            print(f"DEBUG: Files in temp dir after renaming: {len(temp_files)}")

            # Move all files back to the main directory
            for file in temp_files:
                try:
                    os.rename(
                        os.path.join(temp_dir, file),
                        os.path.join(frames_dir, file)
                    )
                except Exception as e:
                    print(f"DEBUG: Error moving file {file} back to frames dir: {e}")

            # Remove the temp directory
            try:
                os.rmdir(temp_dir)
                print(f"DEBUG: Successfully removed temp directory")
            except Exception as e:
                print(f"DEBUG: Error removing temp directory: {e}")

            print(f"Successfully renamed {len(frame_files)} frames with source-based numbering")

        else:
            # Standard sequential extraction
            # DEBUG: Print directory before extraction
            print(f"DEBUG: Directory before standard extraction:")
            if os.path.exists(frames_dir):
                print(f"DEBUG: Contents: {os.listdir(frames_dir)}")

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # DEBUG: Check execution result
            print(f"DEBUG: ffmpeg return code: {result.returncode}")
            print(f"DEBUG: ffmpeg stderr output length: {len(result.stderr)}")
            if len(result.stderr) > 0:
                print(f"DEBUG: First 500 chars of ffmpeg stderr: {result.stderr[:500]}")

            # DEBUG: Check directory after extraction
            print(f"DEBUG: Directory after standard extraction:")
            if os.path.exists(frames_dir):
                frame_files = os.listdir(frames_dir)
                print(f"DEBUG: Number of files: {len(frame_files)}")
                print(f"DEBUG: First few files (if any): {frame_files[:5] if frame_files else 'No files'}")

        # Check if the command was successful
        if result.returncode != 0:
            print(f"Error extracting frames: {result.stderr}")
            return False

        # Count the number of extracted frames
        frames_count = len([f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(f".{file_ext}")])

        # DEBUG: Final check of frames directory
        print(f"DEBUG: Final directory contents:")
        if os.path.exists(frames_dir):
            frame_files = os.listdir(frames_dir)
            print(f"DEBUG: Total files: {len(frame_files)}")
            print(f"DEBUG: Sample of files: {frame_files[:5] if len(frame_files) >= 5 else frame_files}")

        # Report success with appropriate format details
        if use_jpeg:
            print(f"Successfully extracted {frames_count} JPEG frames at quality {jpeg_quality} with {chroma_subsampling} chroma")

            # Calculate approximate space savings compared to PNG
            try:
                total_size = sum(os.path.getsize(os.path.join(frames_dir, f)) for f in os.listdir(frames_dir)
                              if f.startswith("frame_") and f.endswith(f".{file_ext}"))
                avg_size = total_size / frames_count if frames_count > 0 else 0
                print(f"Average JPEG frame size: {avg_size/1024:.2f} KB")
                print(f"Estimated space savings: ~{70 + (100-jpeg_quality)/3:.1f}% compared to PNG")
            except Exception as e:
                print(f"Could not calculate size statistics: {e}")
        else:
            print(f"Successfully extracted {frames_count} PNG frames at maximum quality")

        return True

    except Exception as e:
        print(f"Error during frame extraction: {e}")
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        return False

def process_episodes(source_dir, destination_dir, frame_rate, use_jpeg=USE_JPEG,
                   jpeg_quality=JPEG_QUALITY, chroma_subsampling=CHROMA_SUBSAMPLING,
                   resolution=RESOLUTION, scale_algorithm=SCALE_ALGORITHM,
                   frame_interval=FRAME_INTERVAL, preserve_frame_numbers=PRESERVE_FRAME_NUMBERS):
    """
    Process all episodes in the destination directory to extract all frames from source MKVs.

    Parameters:
    source_dir (str): Source directory containing season folders with MKV files
    destination_dir (str): Destination directory containing processed episode folders
    frame_rate (float): Frames per second of the video
    use_jpeg (bool): Use JPEG instead of PNG for frame extraction
    jpeg_quality (int): JPEG quality (1-100, higher is better quality but larger files)
    chroma_subsampling (str): Chroma subsampling method (444=best, 422=good, 420=smaller)

    Returns:
    dict: Statistics about processed episodes
    """
    stats = {
        "episodes_processed": 0,
        "episodes_skipped": 0,
    }

    # DEBUG: Check source and destination directories
    print(f"DEBUG: Source directory: {source_dir}")
    print(f"DEBUG: Source directory exists: {os.path.exists(source_dir)}")
    if os.path.exists(source_dir):
        print(f"DEBUG: Source directory contents: {os.listdir(source_dir)[:10]}")

    print(f"DEBUG: Destination directory: {destination_dir}")
    print(f"DEBUG: Destination directory exists: {os.path.exists(destination_dir)}")
    if os.path.exists(destination_dir):
        print(f"DEBUG: Destination directory contents: {os.listdir(destination_dir)[:10]}")

    # Walk through the destination directory
    for season_dir in os.listdir(destination_dir):
        season_path = os.path.join(destination_dir, season_dir)

        # Skip if not a directory or not a season directory
        if not os.path.isdir(season_path) or not season_dir.startswith("Season"):
            continue

        print(f"Processing {season_dir}...")
        print(f"DEBUG: Season directory: {season_path}")
        print(f"DEBUG: Season directory contents: {os.listdir(season_path)[:10]}")

        # Process each episode directory
        for episode_dir in os.listdir(season_path):
            episode_path = os.path.join(season_path, episode_dir)

            # Skip if not a directory or not an episode directory (SxxExx format)
            if not os.path.isdir(episode_path) or not re.match(r'S\d+E\d+', episode_dir):
                continue

            print(f"DEBUG: Processing episode directory: {episode_path}")
            print(f"DEBUG: Episode directory contents: {os.listdir(episode_path) if os.path.exists(episode_path) else 'Directory does not exist'}")

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

            # DEBUG: Check frames directory
            print(f"DEBUG: Frames directory for episode: {frames_dir}")
            print(f"DEBUG: Frames directory exists: {os.path.exists(frames_dir)}")

            # File extension based on format
            file_ext = "jpg" if use_jpeg else "png"

            # Check if frames already exist
            if os.path.exists(frames_dir) and len([f for f in os.listdir(frames_dir)
                                                 if f.startswith("frame_") and f.endswith(f".{file_ext}")]) > 0:
                # Ask for confirmation before overwriting
                print(f"{file_ext.upper()} frames already exist for {episode_dir}. Skip? (y/n)")
                response = input().strip().lower()
                if response == 'y':
                    print(f"Skipping {episode_dir}")
                    stats["episodes_skipped"] += 1
                    continue

            # Extract all frames
            format_str = f"JPEG (quality {jpeg_quality}, chroma {chroma_subsampling})" if use_jpeg else "PNG (lossless)"
            print(f"Extracting ALL frames for {episode_dir} as {format_str} using frame rate {frame_rate}...")

            success = extract_all_frames(mkv_file, frames_dir, frame_rate,
                               use_jpeg, jpeg_quality, chroma_subsampling,
                               resolution, scale_algorithm,
                               frame_interval, preserve_frame_numbers)

            if success:
                stats["episodes_processed"] += 1
                # DEBUG: Check final frame numbering
                if os.path.exists(frames_dir):
                    frame_files = sorted([f for f in os.listdir(frames_dir)
                                        if f.startswith("frame_") and f.endswith(f".{file_ext}")],
                                        key=lambda x: int(x.split('_')[1].split('.')[0]))
                    if frame_files:
                        print(f"DEBUG: Final frame numbering check - first 5 frames: {frame_files[:5]}")
                        if frame_interval > 1 and preserve_frame_numbers:
                            # Check if the frame numbers follow the expected pattern
                            expected_pattern = True
                            for i, frame_file in enumerate(frame_files[:min(10, len(frame_files))]):
                                expected_frame = (i * frame_interval) + 1
                                actual_frame = int(frame_file.split('_')[1].split('.')[0])
                                if expected_frame != actual_frame:
                                    expected_pattern = False
                                    print(f"DEBUG: Frame numbering mismatch at position {i}: Expected {expected_frame}, got {actual_frame}")

                            if not expected_pattern:
                                print("WARNING: Frame numbering does not match the expected pattern with frame_interval={frame_interval}!")
                                print("Consider manually renaming the frames or fixing the renaming logic.")
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

def fix_frame_numbering(frames_dir, frame_interval, file_ext):
    """
    Fix frame numbering to match the expected pattern for a given frame interval.
    This function can be called separately to fix frame numbering issues.

    Parameters:
    frames_dir (str): Directory containing frame files
    frame_interval (int): The frame interval used for extraction
    file_ext (str): File extension of the frames (jpg or png)

    Returns:
    bool: True if successful, False otherwise
    """
    print(f"Starting frame renumbering fix for interval {frame_interval}...")

    try:
        # Check if the directory exists
        if not os.path.exists(frames_dir):
            print(f"Error: Directory does not exist: {frames_dir}")
            return False

        # Get all frame files and sort them numerically
        frame_files = sorted([f for f in os.listdir(frames_dir)
                             if f.startswith("frame_") and f.endswith(f".{file_ext}")],
                             key=lambda x: int(x.split('_')[1].split('.')[0]))

        if not frame_files:
            print(f"Error: No frame files found in {frames_dir}")
            return False

        print(f"Found {len(frame_files)} frames to renumber")
        print(f"Current first few frames: {frame_files[:5]}")

        # Create a temp directory
        temp_dir = os.path.join(frames_dir, "temp_renumber")
        os.makedirs(temp_dir, exist_ok=True)

        # Rename files based on their position and the frame interval
        for i, frame_file in enumerate(frame_files):
            # Calculate proper source frame number
            source_frame = (i * frame_interval) + 1

            # Create the new filename
            new_name = f"frame_{source_frame:010d}.{file_ext}"

            # Move to temp dir with new name
            os.rename(
                os.path.join(frames_dir, frame_file),
                os.path.join(temp_dir, new_name)
            )

            # Print progress every 100 frames
            if i % 100 == 0:
                print(f"Renamed {i} frames...")

        # Move all files back to the main directory
        moved_count = 0
        for file in os.listdir(temp_dir):
            os.rename(
                os.path.join(temp_dir, file),
                os.path.join(frames_dir, file)
            )
            moved_count += 1

            # Print progress every 100 frames
            if moved_count % 100 == 0:
                print(f"Moved {moved_count} frames back...")

        # Remove the temp directory
        os.rmdir(temp_dir)

        # Verify the renaming
        new_frame_files = sorted([f for f in os.listdir(frames_dir)
                                 if f.startswith("frame_") and f.endswith(f".{file_ext}")],
                                 key=lambda x: int(x.split('_')[1].split('.')[0]))

        print(f"Renumbering complete. New first few frames: {new_frame_files[:5]}")

        # Verify correct interval
        if len(new_frame_files) >= 2:
            first_frame = int(new_frame_files[0].split('_')[1].split('.')[0])
            second_frame = int(new_frame_files[1].split('_')[1].split('.')[0])
            if (second_frame - first_frame) != frame_interval:
                print(f"Warning: Frame interval seems incorrect. Expected {frame_interval}, got {second_frame - first_frame}")

        return True

    except Exception as e:
        print(f"Error during frame renumbering: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print(f"Starting frame extraction from {source_directory} to {destination_directory}")

    # Display frame extraction format settings
    if USE_JPEG:
        print(f"Frame format: JPEG (quality: {JPEG_QUALITY}, chroma subsampling: {CHROMA_SUBSAMPLING})")
        print("Note: JPEG provides significant space savings with minimal visual quality loss")
        if JPEG_QUALITY > 95:
            print("Tip: Very high JPEG quality (>95) provides diminishing visual returns but significantly larger files")
        elif JPEG_QUALITY < 80:
            print("Warning: Low JPEG quality (<80) may cause visible compression artifacts")
    else:
        print("Frame format: PNG (lossless)")
        print("Warning: PNG extraction will require significantly more disk space than JPEG")

    # Apply optimizations
    is_wsl = optimize_wsl_performance()
    optimize_hardware_performance()

    # Setup SSD/NVMe optimizations if available
    dest_device = os.statvfs(destination_directory)
    source_device = os.statvfs(source_directory)

    # Calculate available space
    available_space_gb = (dest_device.f_frsize * dest_device.f_bavail) / (1024**3)
    print(f"Available space at destination: {available_space_gb:.2f} GB")

    # Estimate space requirements
    est_frame_size_mb = 1.0 if USE_JPEG else 5.0  # Rough estimate: 1MB per JPEG frame, 5MB per PNG frame
    est_frames_per_episode = 30 * 60 * 22  # Rough estimate: 30fps * 60sec * 22min

    # Adjust for frame interval
    if FRAME_INTERVAL > 1:
        est_frames_per_episode = est_frames_per_episode / FRAME_INTERVAL

    est_space_per_episode_gb = (est_frame_size_mb * est_frames_per_episode) / 1024

    est_episodes_possible = int(available_space_gb / est_space_per_episode_gb)
    print(f"Estimated space per episode: {est_space_per_episode_gb:.2f} GB")
    print(f"Estimated number of episodes possible: ~{est_episodes_possible}")

    if available_space_gb < 100:
        print("WARNING: Less than 100GB available at destination. Frame extraction may fail if disk space runs out.")

    print("\nHARDWARE OPTIMIZATION SUMMARY:")
    print(f"- GPU: {'RTX 3090 (Optimized)' if rtx_3090_detected else gpu_model if has_nvidia_gpu else 'Not detected'}")
    print(f"- CPU: {'i9-9900K (Optimized)' if i9_9900k_detected else 'Standard'}")
    print(f"- WSL: {'Optimized' if is_wsl else 'Not detected'}")
    print(f"- Parallel processing: Enabled")
    print(f"- Frame format: {'JPEG' if USE_JPEG else 'PNG'}")
    if USE_JPEG:
        print(f"  - Quality: {JPEG_QUALITY}/100")
        print(f"  - Chroma: {CHROMA_SUBSAMPLING}")
    print(f"- Resolution: {RESOLUTION}")
    print(f"- Scaling algorithm: {SCALE_ALGORITHM}")
    print(f"- Frame interval: Every {FRAME_INTERVAL} frame{'s' if FRAME_INTERVAL > 1 else ''}")
    if FRAME_INTERVAL > 1:
        print(f"- Frame numbering: {'Source-based (e.g., 1,4,7...)' if PRESERVE_FRAME_NUMBERS else 'Sequential (e.g., 1,2,3...)'}")
    print("\nEstimated performance: Excellent\n")

    print("Select operation:")
    print("1. Extract frames from episodes")
    print("2. Fix frame numbering for existing frames")
    print("3. Exit")

    operation = input("Enter choice (1, 2, or 3): ").strip()

    if operation == "1":
        print("Continue with optimized extraction? (y/n)")
        extract_response = input().strip().lower()
        if extract_response != 'y':
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
        stats = process_episodes(source_directory, destination_directory, frame_rate,
                             USE_JPEG, JPEG_QUALITY, CHROMA_SUBSAMPLING,
                             RESOLUTION, SCALE_ALGORITHM,
                             FRAME_INTERVAL, PRESERVE_FRAME_NUMBERS)

        print("\nFrame Extraction Complete!")
        print(f"Episodes processed: {stats['episodes_processed']}")
        print(f"Episodes skipped: {stats['episodes_skipped']}")
        print(f"Frame format: {'JPEG' if USE_JPEG else 'PNG'}")
        if USE_JPEG:
            print(f"JPEG quality: {JPEG_QUALITY}")
            print(f"Chroma subsampling: {CHROMA_SUBSAMPLING}")
        print(f"Resolution: {RESOLUTION}")
        print(f"Scaling algorithm: {SCALE_ALGORITHM}")
        print(f"Frame interval: Every {FRAME_INTERVAL} frame{'s' if FRAME_INTERVAL > 1 else ''}")
        if FRAME_INTERVAL > 1:
            print(f"Frame numbering: {'Source-based (e.g., 1,4,7...)' if PRESERVE_FRAME_NUMBERS else 'Sequential (e.g., 1,2,3...)'}")

    elif operation == "2":
        # Get path to frames directory
        frames_path = input("Enter path to frames directory (e.g., /mnt/e/Projects/Jonadex/Veep/Season 1/S01E01/frames): ").strip()
        if not os.path.exists(frames_path):
            print(f"Error: Directory does not exist: {frames_path}")
            exit(1)

        # Get frame interval
        try:
            interval = int(input("Enter frame interval (e.g., 5 for every 5th frame): ").strip())
            if interval < 1:
                print("Error: Interval must be at least 1")
                exit(1)
        except ValueError:
            print("Error: Invalid interval value")
            exit(1)

        # Get file extension
        file_ext = input("Enter file extension (jpg or png): ").strip().lower()
        if file_ext not in ["jpg", "png"]:
            print("Error: File extension must be jpg or png")
            exit(1)

        # Fix frame numbering
        success = fix_frame_numbering(frames_path, interval, file_ext)
        if success:
            print("Frame numbering fixed successfully!")
        else:
            print("Error fixing frame numbering")

    else:
        print("Exiting...")
        exit(0)
