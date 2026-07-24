import sqlite3
import json
import os
from pathlib import Path

def create_database(json_dir, db_path="wikipedia_local.db"):
    print(f"Creating database at {db_path}...")
    
    # Connect to SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Optimize SQLite for massive bulk inserts
    cursor.execute("PRAGMA journal_mode=OFF;")
    cursor.execute("PRAGMA synchronous=0;")
    cursor.execute("PRAGMA cache_size=1000000;")
    cursor.execute("PRAGMA locking_mode=EXCLUSIVE;")
    
    # Create the articles table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT COLLATE NOCASE,
            text TEXT
        )
    ''')
    
    print("Reading JSON files from WikiExtractor...")
    
    # Find all the extracted JSON files
    path = Path(json_dir)
    json_files = list(path.rglob("wiki_*"))
    
    total_files = len(json_files)
    articles_batch = []
    batch_size = 10000
    total_inserted = 0
    
    for i, file_path in enumerate(json_files):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    # We only need title and text for the WikiEnv
                    articles_batch.append((data['title'], data['text']))
                    
                    if len(articles_batch) >= batch_size:
                        cursor.executemany(
                            "INSERT INTO articles (title, text) VALUES (?, ?)", 
                            articles_batch
                        )
                        total_inserted += len(articles_batch)
                        articles_batch = []
                except json.JSONDecodeError:
                    continue
        
        # Print progress
        if i % 10 == 0 or i == total_files - 1:
            print(f"Processed {i+1}/{total_files} files... ({total_inserted} articles inserted)")
            
    # Insert any remaining articles
    if articles_batch:
        cursor.executemany(
            "INSERT INTO articles (title, text) VALUES (?, ?)", 
            articles_batch
        )
        total_inserted += len(articles_batch)
        
    print(f"Total articles inserted: {total_inserted}")
    print("Creating index on 'title' for fast searching (this may take a few minutes)...")
    
    # Create an index on the title column. 
    # This makes the search_step() exact match query almost instantaneous.
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_title ON articles(title)")
    
    conn.commit()
    conn.close()
    
    print("Database built successfully!")

if __name__ == "__main__":
    # Point this to the output directory of WikiExtractor
    create_database("/mnt/scratch/nunes/wiki/wiki_extracted", "wikipedia_local.db")