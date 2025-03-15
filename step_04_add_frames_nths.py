import os
import pandas as pd
import json
from showenv import destination_directory

# Import frame_rate from showenv, with a fallback default value
try:
    from showenv import frame_rate
    print(f"Using frame rate from showenv: {frame_rate} fps")
except ImportError:
    frame_rate = 23.976
    print(f"Frame rate not found in showenv, using default: {frame_rate} fps")

def timestamp_to_seconds(timestamp):
    """
    Convert a timestamp in format HH:MM:SS,MS to seconds.

    Parameters:
    timestamp (str): Timestamp in format HH:MM:SS,MS

    Returns:
    float: Time in seconds
    """
    # Parse the timestamp
    hours, minutes, seconds = timestamp.split(':')
    seconds, milliseconds = seconds.split(',')

    # Convert to seconds
    total_seconds = (
        int(hours) * 3600 +
        int(minutes) * 60 +
        int(seconds) +
        int(milliseconds) / 1000
    )

    return total_seconds

def seconds_to_frame(seconds, frame_rate):
    """
    Convert seconds to frame number.

    Parameters:
    seconds (float): Time in seconds
    frame_rate (float): Frames per second

    Returns:
    int: Frame number
    """
    return int(seconds * frame_rate)

def add_all_frames_to_csv(csv_file, frame_rate):
    """
    Add a column to the CSV with all frame numbers as a JSON array.

    Parameters:
    csv_file (str): Path to the CSV file
    frame_rate (float): Frames per second

    Returns:
    bool: True if successful, False otherwise
    """
    try:
        # Check if file exists
        if not os.path.exists(csv_file):
            print(f"Warning: CSV file not found at {csv_file}")
            return False

        # Read CSV file
        df = pd.read_csv(csv_file)

        # Check if the file has the required columns
        if 'Timestamp' not in df.columns:
            print(f"Warning: CSV file {csv_file} does not have a Timestamp column")
            return False

        # Add new columns for frame information
        df['Start Frame'] = 0
        df['End Frame'] = 0
        df['All Frames'] = None

        # Process each row
        for index, row in df.iterrows():
            timestamp = row['Timestamp']

            # Extract start and end timestamps
            timestamps = timestamp.split(' --> ')
            if len(timestamps) != 2:
                print(f"Warning: Invalid timestamp format in row {index}: {timestamp}")
                continue

            start_timestamp, end_timestamp = timestamps

            # Convert timestamps to seconds
            start_seconds = timestamp_to_seconds(start_timestamp)
            end_seconds = timestamp_to_seconds(end_timestamp)

            # Convert seconds to frame numbers
            start_frame = seconds_to_frame(start_seconds, frame_rate)
            end_frame = seconds_to_frame(end_seconds, frame_rate)

            # Update the frame columns
            df.at[index, 'Start Frame'] = start_frame
            df.at[index, 'End Frame'] = end_frame

            # Generate all frames in range
            all_frames = list(range(start_frame, end_frame + 1))

            # Store as JSON array
            df.at[index, 'All Frames'] = json.dumps(all_frames)

        # Save the updated CSV (overwrite existing file)
        df.to_csv(csv_file, index=False)

        print(f"Updated {csv_file} with frame information")
        return True

    except Exception as e:
        print(f"Error updating {csv_file} with frame information: {e}")
        return False

def process_all_csv_files(destination_dir, frame_rate):
    """
    Walk through the destination directory and update all CSV files with frame information.

    Parameters:
    destination_dir (str): The root directory containing the organized subtitle files
    frame_rate (float): Frames per second

    Returns:
    tuple: (Number of files successfully updated, total number of CSV files found)
    """
    success_count = 0
    total_count = 0

    # Walk through all directories under the destination directory
    for root, dirs, files in os.walk(destination_dir):
        for file in files:
            if file == 'subtitles.csv':
                total_count += 1
                csv_path = os.path.join(root, file)

                # Update CSV with frame information
                if add_all_frames_to_csv(csv_path, frame_rate):
                    success_count += 1

    return success_count, total_count

if __name__ == "__main__":
    print(f"Starting update of CSV files with frame information in {destination_directory}")

    # Process all CSV files
    success_count, total_count = process_all_csv_files(destination_directory, frame_rate)

    print(f"Update completed. Successfully updated {success_count} of {total_count} CSV files with frame information.")
