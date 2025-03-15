import os
import re
import pandas as pd
from showenv import destination_directory

def parse_srt(srt_file):
    """
    Parse an SRT file and return a pandas DataFrame of subtitle entries.

    Handles malformed SRT files where the subtitle number might appear at the
    end of the previous subtitle's content.

    Parameters:
    srt_file (str): Path to the SRT file.

    Returns:
    pandas.DataFrame: DataFrame containing subtitle information.
    """
    with open(srt_file, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # First, let's identify all timestamps
    timestamp_pattern = r'(\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3})'

    # Split the content by timestamps
    parts = re.split(timestamp_pattern, content)

    # Initialize lists to store our data
    numbers = []
    timestamps = []
    contents = []

    # Process parts in groups of 3: before timestamp, timestamp, after timestamp
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            # The timestamp itself
            timestamp = parts[i]

            # The content before the timestamp (should contain the subtitle number)
            before = parts[i-1].strip()

            # The content after the timestamp
            after = parts[i+1].strip()

            # Extract subtitle number from the content before timestamp
            number_match = re.search(r'(\d+)\s*$', before)
            if number_match:
                subtitle_number = number_match.group(1)
            else:
                # If we can't find a number, use a placeholder
                subtitle_number = f"Unknown-{i//2}"

            # Clean and extract the content
            # Check if the content contains the next subtitle number
            next_number_match = re.search(r'\s(\d+)\s*$', after)
            if next_number_match:
                # Remove the next subtitle number from the content
                content_text = after[:next_number_match.start()]
            else:
                content_text = after

            # Clean up the content text (join lines with spaces)
            content_text = ' '.join(line.strip() for line in content_text.splitlines())

            numbers.append(subtitle_number)
            timestamps.append(timestamp)
            contents.append(content_text)

    # Create a DataFrame
    return pd.DataFrame({
        'Number': numbers,
        'Timestamp': timestamps,
        'Content': contents
    })

def convert_srt_to_csv(srt_file, csv_file):
    """
    Convert an SRT file to a CSV file using pandas.

    Parameters:
    srt_file (str): Path to the SRT file.
    csv_file (str): Path where the CSV file will be saved.

    Returns:
    bool: True if successful, False otherwise.
    """
    try:
        # Parse the SRT file into a DataFrame
        df = parse_srt(srt_file)

        if df.empty:
            print(f"Warning: No subtitles found in {srt_file}")
            return False

        # Write DataFrame to CSV
        df.to_csv(csv_file, index=False, encoding='utf-8')

        print(f"Converted {srt_file} to {csv_file}")
        return True

    except Exception as e:
        print(f"Error converting {srt_file} to CSV: {e}")
        return False

def process_all_srt_files(destination_dir):
    """
    Walk through the destination directory and convert all SRT files to CSV.

    Parameters:
    destination_dir (str): The root directory containing the organized subtitle files.

    Returns:
    tuple: Number of files successfully converted and total number of SRT files.
    """
    success_count = 0
    total_count = 0

    # Walk through all directories under the destination directory
    for root, dirs, files in os.walk(destination_dir):
        for file in files:
            if file.endswith('.srt'):
                total_count += 1
                srt_path = os.path.join(root, file)

                # Create CSV file path
                csv_path = os.path.join(root, 'subtitles.csv')

                # Convert SRT to CSV
                if convert_srt_to_csv(srt_path, csv_path):
                    success_count += 1

    return success_count, total_count

if __name__ == "__main__":
    print(f"Starting conversion of SRT files to CSV in {destination_directory}")

    # Process all SRT files
    success_count, total_count = process_all_srt_files(destination_directory)

    print(f"Conversion completed. Successfully converted {success_count} of {total_count} SRT files to CSV.")
