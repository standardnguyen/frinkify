import os
import re
import sys
from showenv import source_directory, destination_directory


def parse_episode_info(filename):
    """
    Parse season, episode, and title information from filename.
    Example: "Veep (2012) - S01E01 - Fundraiser (1080p BluRay x265 Silence).mkv"

    Parameters:
    filename (str): The filename to parse

    Returns:
    tuple: (season_number, episode_number, episode_title) or None if parsing fails
    """
    # Extract season and episode using regex
    match = re.search(r'S(\d+)E(\d+)\s*-\s*([^(]+)', filename)
    if match:
        season_num = int(match.group(1))
        episode_num = int(match.group(2))
        episode_title = match.group(3).strip()
        return season_num, episode_num, episode_title
    return None

def find_all_mkv_files(source_dir):
    """
    Find all MKV files in the source directory structure.

    Parameters:
    source_dir (str): Source directory containing season folders

    Returns:
    list: List of tuples (filepath, season, episode, title)
    """
    all_episodes = []

    # Search recursively through the source directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".mkv"):
                filepath = os.path.join(root, file)
                info = parse_episode_info(file)
                if info:
                    season_num, episode_num, episode_title = info
                    all_episodes.append((filepath, season_num, episode_num, episode_title))
                    print(f"Found S{season_num:02d}E{episode_num:02d}: {episode_title}")
                else:
                    print(f"Warning: Could not parse episode info from {file}")

    return all_episodes

def create_title_files(episodes, destination_dir):
    """
    Create title.txt files in each episode directory.

    Parameters:
    episodes (list): List of tuples (filepath, season, episode, title)
    destination_dir (str): Destination directory containing processed episode folders

    Returns:
    int: Number of title files created
    """
    count = 0

    for _, season_num, episode_num, episode_title in episodes:
        # Construct the path to the episode directory
        season_dir = os.path.join(destination_dir, f"Season {season_num}")
        episode_dir = os.path.join(season_dir, f"S{season_num:02d}E{episode_num:02d}")

        # Check if episode directory exists
        if not os.path.exists(episode_dir):
            print(f"Creating directory: {episode_dir}")
            os.makedirs(episode_dir, exist_ok=True)

        # Create title.txt file
        title_file = os.path.join(episode_dir, "title.txt")
        with open(title_file, 'w', encoding='utf-8') as f:
            f.write(episode_title)

        print(f"Created title file for S{season_num:02d}E{episode_num:02d}: {episode_title}")
        count += 1

    return count

if __name__ == "__main__":
    print(f"Starting episode title extraction from {source_directory}")
    print(f"Saving titles to {destination_directory}")

    # Find all episodes
    all_episodes = find_all_mkv_files(source_directory)

    if not all_episodes:
        print("No episodes found. Please check the source directory.")
        sys.exit(1)

    print(f"Found {len(all_episodes)} episodes.")

    # Ask for confirmation
    print("\nCreate title.txt files for these episodes? (y/n)")
    response = input().strip().lower()
    if response != 'y':
        print("Title extraction cancelled.")
        sys.exit(0)

    # Create title files
    count = create_title_files(all_episodes, destination_directory)

    print(f"\nTitle Extraction Complete!")
    print(f"Created {count} title.txt files")
