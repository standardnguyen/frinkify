import os
import re
from showenv import destination_directory

def clean_html_from_srt(input_file, output_file):
    """
    Remove HTML tags from an SRT file.

    Parameters:
    input_file (str): Path to the original SRT file.
    output_file (str): Path where the cleaned SRT file will be saved.
    """
    print(f"Cleaning HTML from {input_file}")

    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Remove HTML tags
    clean_content = re.sub(r'<[^>]+>', '', content)

    # Fix any double spaces that might be left after removing tags
    clean_content = re.sub(r'\s{2,}', ' ', clean_content)

    # Remove any font tags that might be left
    clean_content = re.sub(r'\{\\[^}]+\}', '', clean_content)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(clean_content)

    print(f"Cleaned file saved to {output_file}")

def process_all_srt_files(destination_dir):
    """
    Walk through the destination directory and clean all SRT files.

    Parameters:
    destination_dir (str): The root directory containing the organized subtitle files.
    """
    counter = 0

    # Walk through all directories under the destination directory
    for root, dirs, files in os.walk(destination_dir):
        for file in files:
            if file.endswith('.srt'):
                srt_path = os.path.join(root, file)

                # Create a temporary file path for the cleaned version
                temp_file = srt_path + '.temp'

                try:
                    # Clean the SRT file
                    clean_html_from_srt(srt_path, temp_file)

                    # Replace the original file with the cleaned one
                    os.replace(temp_file, srt_path)
                    counter += 1
                except Exception as e:
                    print(f"Error processing {srt_path}: {e}")
                    # Clean up temp file if it exists
                    if os.path.exists(temp_file):
                        os.remove(temp_file)

    return counter

if __name__ == "__main__":
    print(f"Starting HTML cleanup of SRT files in {destination_directory}")

    # Process all SRT files
    processed_count = process_all_srt_files(destination_directory)

    print(f"HTML cleanup completed. Processed {processed_count} SRT files.")
