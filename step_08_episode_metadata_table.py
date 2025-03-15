import os
import sqlite3
import re
import sys

# Import configuration from showenv
try:
    from showenv import destination_directory
    print(f"Using destination directory from showenv: {destination_directory}")
except ImportError:
    destination_directory = "./output"  # Default fallback
    print(f"Destination directory not found in showenv, using default: {destination_directory}")

def create_episodes_table(conn):
    """
    Create a new episodes table in the SQLite database.

    Parameters:
    conn (sqlite3.Connection): Connection to the database

    Returns:
    bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        # Create episodes table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            episode_of_season INTEGER NOT NULL,
            episode_overall INTEGER NOT NULL,
            title TEXT NOT NULL,
            UNIQUE(season, episode_of_season)
        )
        ''')

        # Create indices for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_episodes_season ON episodes (season)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_episodes_title ON episodes (title)')

        # Commit changes
        conn.commit()
        return True

    except Exception as e:
        print(f"Error creating episodes table: {e}")
        conn.rollback()
        return False

def find_all_title_files(destination_dir):
    """
    Find all title.txt files in the destination directory structure.

    Parameters:
    destination_dir (str): Root directory to search in

    Returns:
    list: List of tuples (season, episode, path_to_title_file)
    """
    title_files = []

    # Walk through all directories under the destination directory
    for root, dirs, files in os.walk(destination_dir):
        if "title.txt" in files:
            # Extract season and episode from directory path
            path_parts = os.path.normpath(root).split(os.sep)

            # Look for SxxExx pattern in the last directory part
            season_episode_match = re.search(r'S(\d+)E(\d+)', path_parts[-1])

            if season_episode_match:
                season = int(season_episode_match.group(1))
                episode = int(season_episode_match.group(2))
                title_path = os.path.join(root, "title.txt")
                title_files.append((season, episode, title_path))
                print(f"Found title file for S{season:02d}E{episode:02d} at {title_path}")

    # Sort by season and episode
    title_files.sort()
    return title_files

def calculate_overall_episode_numbers(title_files):
    """
    Calculate the overall episode number for each episode.

    Parameters:
    title_files (list): List of tuples (season, episode, path_to_title_file)

    Returns:
    list: List of tuples (season, episode, overall_episode, path_to_title_file)
    """
    # Group by season
    seasons = {}
    for season, episode, path in title_files:
        if season not in seasons:
            seasons[season] = []
        seasons[season].append((episode, path))

    # Sort each season's episodes
    for season in seasons:
        seasons[season].sort()

    # Calculate overall episode number
    result = []
    overall_episode = 1

    for season in sorted(seasons.keys()):
        for episode, path in seasons[season]:
            result.append((season, episode, overall_episode, path))
            overall_episode += 1

    return result

def populate_episodes_table(conn, episodes_info):
    """
    Populate the episodes table with data from title.txt files.

    Parameters:
    conn (sqlite3.Connection): Connection to the database
    episodes_info (list): List of tuples (season, episode_of_season, episode_overall, path_to_title_file)

    Returns:
    int: Number of episodes added to the table
    """
    cursor = conn.cursor()
    count = 0

    for season, episode_of_season, episode_overall, title_path in episodes_info:
        try:
            # Read the title from the file
            with open(title_path, 'r', encoding='utf-8') as f:
                title = f.read().strip()

            # Insert episode data
            cursor.execute('''
            INSERT INTO episodes (season, episode_of_season, episode_overall, title)
            VALUES (?, ?, ?, ?)
            ''', (season, episode_of_season, episode_overall, title))

            count += 1
            print(f"Added S{season:02d}E{episode_of_season:02d} (#{episode_overall}): {title}")

        except Exception as e:
            print(f"Error processing {title_path}: {e}")

    # Commit changes
    conn.commit()
    return count

def main():
    # Define database path
    db_path = os.path.join(destination_directory, "subtitles.db")

    # Check if the database exists
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        print("Please run Step 06 first to create the database.")
        return False

    # Connect to the database
    print(f"Connecting to SQLite database at {db_path}")
    conn = sqlite3.connect(db_path)

    # Create episodes table
    print("Creating episodes table...")
    if not create_episodes_table(conn):
        print("Failed to create episodes table. Exiting.")
        conn.close()
        return False

    # Find all title.txt files
    print(f"Searching for title.txt files in {destination_directory}...")
    title_files = find_all_title_files(destination_directory)

    if not title_files:
        print("No title.txt files found. Please run Step 07 first.")
        conn.close()
        return False

    print(f"Found {len(title_files)} title files.")

    # Calculate overall episode numbers
    print("Calculating overall episode numbers...")
    episodes_info = calculate_overall_episode_numbers(title_files)

    # Populate the episodes table
    print("Populating episodes table...")
    count = populate_episodes_table(conn, episodes_info)

    # Print summary
    print(f"\nEpisodes Table Creation Complete!")
    print(f"Added {count} episodes to the database.")

    # Add some example queries
    print("\nExample queries you can run with the episodes table:")
    print("\n1. View all episodes:")
    print("   SELECT * FROM episodes ORDER BY season, episode_of_season;")
    print("\n2. Get episode information by title:")
    print("   SELECT * FROM episodes WHERE title LIKE '%keyword%';")
    print("\n3. Join with subtitles to find dialogue in specific episodes:")
    print("   SELECT e.title, s.content FROM episodes e JOIN subtitles s")
    print("   ON e.season = s.season AND e.episode_of_season = s.episode")
    print("   WHERE s.content LIKE '%search term%';")

    # Close the database connection
    conn.close()
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
