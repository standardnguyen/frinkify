import os
import re
import csv
from showenv import destination_directory

def parse_srt(srt_file):
    """
    Parse an SRT file and return a list of subtitle entries.

    Each entry is a dictionary with:
    - number: the subtitle block number
    - timestamp: the formatted timestamp range
    - content: the subtitle text with line breaks removed

    Parameters:
    srt_file (str): Path to the SRT file.

    Returns:
    list: List of dictionaries containing subtitle information.
    """
    with open(srt_file, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Pattern to match subtitle blocks
    # Group 1: subtitle number
    # Group 2: timestamp
    # Group 3: subtitle text
    pattern = r'(\d+)\s*\n(\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3})\s*\n((?:.+\n?)+?)(?:\n\n|\Z)'

    entries = []
    for match in re.finditer(pattern, content, re.DOTALL):
        number = match.group(1)
        timestamp = match.group(2)

        # Get subtitle text and clean it
        text = match.group(3)
        # Remove any extra whitespace and join lines
        text = ' '.join(line.strip() for line in text.splitlines())

        entries.append({
            'number': number,
            'timestamp': timestamp,
            'content': text
        })

    return entries

def convert_srt_to_csv(srt_file, csv_file):
    """
    Convert an SRT file to a CSV file.

    Parameters:
    srt_file (str): Path to the SRT file.
    csv_file (str): Path where the CSV file will be saved.

    Returns:
    bool: True if successful, False otherwise.
    """
    try:
        # Parse the SRT file
        entries = parse_srt(srt_file)

        if not entries:
            print(f"Warning: No subtitles found in {srt_file}")
            return False

        # Write to CSV
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['Number', 'Timestamp', 'Content'])

            # Write data
            for entry in entries:
                writer.writerow([
                    entry['number'],
                    entry['timestamp'],
                    entry['content']
                ])

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
    int: Number of files successfully converted.
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
