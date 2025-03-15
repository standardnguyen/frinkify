import os
import re
import shutil
from showenv import destination_directory, subtitles_directory

def copy_subtitles(subtitles_dir, destination_dir):
    """
    Process all subtitle files in the subtitles directory and copy them to the destination directory.

    Parameters:
    subtitles_dir (str): The directory containing the subtitle files organized by season/episode.
    destination_dir (str): The directory where the subtitles should be copied to.
    """
    # Define the range of seasons to process
    # Get all directories in the subtitles directory that start with "Season "
    season_dirs = [d for d in os.listdir(subtitles_dir) if os.path.isdir(os.path.join(subtitles_dir, d)) and d.startswith("Season ")]
    seasons = sorted(season_dirs)

    print(f"Found the following seasons: {seasons}")

    # Loop through each season folder
    for season in seasons:
        season_path = os.path.join(subtitles_dir, season)

        # Create corresponding destination directory for the season
        destination_season_path = os.path.join(destination_dir, season)
        os.makedirs(destination_season_path, exist_ok=True)

        # Get all episode directories in the current season
        episode_dirs = [d for d in os.listdir(season_path) if os.path.isdir(os.path.join(season_path, d))]

        print(f"Processing {season}, found {len(episode_dirs)} episode directories")

        # Process each episode directory
        for episode_dir in sorted(episode_dirs):
            episode_path = os.path.join(season_path, episode_dir)

            # Try to extract episode number for destination folder name
            pattern = r"Episode (\d+)"
            match = re.search(pattern, episode_dir)

            if match:
                episode_num = int(match.group(1))
                season_num = int(season.split(" ")[1])

                # Format destination directory name like "S01E01 - Episode Title"
                formatted_name = f"S{season_num:02d}E{episode_num:02d}"
                destination_episode_path = os.path.join(destination_season_path, formatted_name)
            else:
                # If pattern not found, use the original directory name
                print(f"Warning: Could not extract episode number from directory: {episode_dir}")
                destination_episode_path = os.path.join(destination_season_path, episode_dir)

            # Create the destination episode directory if it doesn't exist
            os.makedirs(destination_episode_path, exist_ok=True)

            # Find SRT files in the episode directory
            srt_files = [f for f in os.listdir(episode_path) if f.endswith('.srt')]

            if srt_files:
                # Take the first SRT file found (assuming there's only one per episode)
                source_srt = os.path.join(episode_path, srt_files[0])
                destination_srt = os.path.join(destination_episode_path, "subtitles.srt")

                # Copy the SRT file to the destination with the standardized name
                shutil.copy2(source_srt, destination_srt)
                print(f"Copied subtitle: {source_srt} -> {destination_srt}")
            else:
                print(f"No SRT files found in {episode_path}")

if __name__ == "__main__":
    print(f"Subtitles directory: {subtitles_directory}")
    print(f"Destination directory: {destination_directory}")

    # Call the function to copy subtitles
    copy_subtitles(subtitles_directory, destination_directory)

    print("Subtitle copying completed.")
