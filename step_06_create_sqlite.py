import os
import pandas as pd
import sqlite3
import re

# Import configuration from showenv
try:
    from showenv import destination_directory
    print(f"Using destination directory from showenv: {destination_directory}")
except ImportError:
    destination_directory = "./output"  # Default fallback
    print(f"Destination directory not found in showenv, using default: {destination_directory}")

def create_database(db_path):
    """
    Create a new SQLite database with just the subtitles table.

    Parameters:
    db_path (str): Path to the SQLite database file

    Returns:
    sqlite3.Connection: Connection to the database
    """
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Connect to the database (will create it if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create subtitles table with episode and season columns as integers
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS subtitles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        episode INTEGER NOT NULL,
        season INTEGER NOT NULL,
        file_path TEXT NOT NULL,
        subtitle_number INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        timestamp_start TEXT NOT NULL,
        timestamp_end TEXT NOT NULL,
        content TEXT NOT NULL,
        start_frame INTEGER NOT NULL,
        end_frame INTEGER NOT NULL
    )
    ''')

    # Create indices for faster queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_subtitles_episode ON subtitles (episode)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_subtitles_season ON subtitles (season)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_subtitles_content ON subtitles (content)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_subtitles_frames ON subtitles (start_frame, end_frame)')

    # Commit changes
    conn.commit()

    return conn

def extract_episode_season(file_path):
    """
    Extract episode and season information from the folder structure as integers.

    Parameters:
    file_path (str): Path to the CSV file

    Returns:
    tuple: (episode, season) as integers
    """
    # Get directory parts
    path_parts = os.path.normpath(file_path).split(os.sep)

    # Default values
    episode = 0
    season = 0

    # Try to find season/episode in directory names
    for part in path_parts:
        # Look for season pattern (e.g., S01, Season1, etc.)
        season_match = re.search(r'(?:s|season)[_\s]*(\d+)', part, re.IGNORECASE)
        if season_match:
            season = int(season_match.group(1))

        # Look for episode pattern (e.g., E01, Episode1, etc.)
        episode_match = re.search(r'(?:e|episode)[_\s]*(\d+)', part, re.IGNORECASE)
        if episode_match:
            episode = int(episode_match.group(1))

    # If we couldn't find episode pattern, check for just numeric values in directory names
    if episode == 0:
        for part in path_parts:
            # Try to find standalone numbers that might represent episodes
            num_match = re.search(r'^(\d+)$', part)
            if num_match:
                try:
                    episode = int(num_match.group(1))
                    break
                except ValueError:
                    pass

    return episode, season

def process_csv_to_db(conn, csv_path):
    """
    Process a CSV file and insert its data into the database.

    Parameters:
    conn (sqlite3.Connection): Connection to the database
    csv_path (str): Path to the CSV file

    Returns:
    bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        # Extract episode and season from file path
        episode, season = extract_episode_season(csv_path)

        # Read CSV file
        df = pd.read_csv(csv_path)

        # Insert each subtitle entry
        for _, row in df.iterrows():
            subtitle_number = row['Number'] if 'Number' in df.columns else 0
            timestamp = row['Timestamp']
            content = row['Content']
            start_frame = row['Start Frame']
            end_frame = row['End Frame']

            # Extract timestamp_start and timestamp_end
            timestamps = timestamp.split(' --> ')
            if len(timestamps) != 2:
                print(f"Warning: Invalid timestamp format in row {subtitle_number}: {timestamp}")
                continue

            timestamp_start, timestamp_end = timestamps

            # Insert subtitle entry
            cursor.execute('''
            INSERT INTO subtitles (
                episode, season, file_path, subtitle_number, timestamp,
                timestamp_start, timestamp_end, content, start_frame, end_frame
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                episode, season, csv_path, subtitle_number, timestamp,
                timestamp_start, timestamp_end, content, start_frame, end_frame
            ))

        # Commit changes
        conn.commit()
        return True

    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        conn.rollback()  # Rollback changes if there was an error
        return False

def process_all_csv_files(conn, destination_dir):
    """
    Walk through the destination directory and process all CSV files.

    Parameters:
    conn (sqlite3.Connection): Connection to the database
    destination_dir (str): The root directory containing the organized subtitle files

    Returns:
    tuple: (Number of files successfully processed, total number of CSV files found)
    """
    success_count = 0
    total_count = 0

    # Walk through all directories under the destination directory
    for root, dirs, files in os.walk(destination_dir):
        for file in files:
            if file == 'subtitles.csv':
                total_count += 1
                csv_path = os.path.join(root, file)

                print(f"Processing {csv_path}...")

                # Process CSV file
                if process_csv_to_db(conn, csv_path):
                    success_count += 1

    return success_count, total_count

if __name__ == "__main__":
    # Define database path
    db_path = os.path.join(destination_directory, "subtitles.db")

    # Remove existing database file if it exists
    if os.path.exists(db_path):
        print(f"Removing existing database at {db_path}")
        os.remove(db_path)

    print(f"Creating new SQLite database at {db_path}")
    conn = create_database(db_path)

    print(f"Starting to process CSV files in {destination_directory}")
    success_count, total_count = process_all_csv_files(conn, destination_directory)

    print(f"Database creation completed. Successfully processed {success_count} of {total_count} CSV files.")

    # Add some example queries
    print("\nExample queries you can run with the database:")
    print("\n1. Search for subtitles containing specific text:")
    print("   SELECT * FROM subtitles WHERE content LIKE '%search term%';")
    print("\n2. Get subtitles for a specific episode:")
    print("   SELECT * FROM subtitles WHERE episode = 5;")
    print("\n3. Get subtitles within a range of frames:")
    print("   SELECT * FROM subtitles WHERE start_frame <= 1000 AND end_frame >= 1000;")
    print("\n4. Get subtitles from a specific timestamp range:")
    print("   SELECT * FROM subtitles WHERE timestamp_start >= '00:10:00,000' AND timestamp_end <= '00:15:00,000';")

    # Close the database connection
    conn.close()
