import sqlite3
import json
from pathlib import Path

DB_FILE = "../data/expenses.db"
JSON_FILE = "../data/extracted_receipts.json"

def create_database():
    """Connects to SQLite and creates the table schema."""
    print(f"Connecting to database: {DB_FILE}")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create a structured table for data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT UNIQUE,
            merchant TEXT,
            purchase_date TEXT,
            total_amount REAL,
            category TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    return conn

def clean_amount(amount_str):
    """Attempts to convert OCR text to a clean float. Returns None if it fails."""
    try:
        # Remove any stray spaces or currency symbols if they sneaked in
        clean_str = str(amount_str).replace('$', '').replace(',', '').strip()
        return float(clean_str)
    except (ValueError, TypeError):
        return None

def load_data_to_db(conn):
    """Reads the JSON file and inserts records into the database."""
    print(f"Reading data from {JSON_FILE}...")
    
    if not Path(JSON_FILE).exists():
        print("JSON file not found! Wait for main.py to finish.")
        return

    with open(JSON_FILE, 'r') as f:
        data = json.load(f)

    cursor = conn.cursor()
    inserted_count = 0

    for item in data:
        # Clean the total amount before inserting
        clean_total = clean_amount(item.get('total_amount'))
        
        try:
            # Insert data. 
            cursor.execute('''
                INSERT OR IGNORE INTO receipts 
                (image_name, merchant, purchase_date, total_amount, category, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                item.get('image_name'),
                item.get('merchant'),
                item.get('date'),
                clean_total,
                item.get('category'),
                item.get('category_confidence')
            ))
            
            # If a new row was actually inserted, increment our counter
            if cursor.rowcount > 0:
                inserted_count += 1
                
        except sqlite3.Error as e:
            print(f"Database error on {item.get('image_name')}: {e}")

    conn.commit()
    print(f"Successfully inserted {inserted_count} new records into the database.")

def run_sample_query(conn):
    print("\n--- Running Cleaned Analytics Query (Under 20k MYR) ---")
    cursor = conn.cursor()
    
    # Query 1: Cleaned spending summary using SQL to filter out phone numbers
    cursor.execute('''
        SELECT category, ROUND(SUM(total_amount), 2) as total_spent, COUNT(*) as receipt_count
        FROM receipts
        WHERE total_amount IS NOT NULL 
        AND total_amount < 20000  
        GROUP BY category
        ORDER BY total_spent DESC
    ''')
    
    results = cursor.fetchall()
    print(f"{'CATEGORY':<35} | {'TOTAL SPENT':<15} | {'RECEIPTS'}")
    print("-" * 65)
    for row in results:
        print(f"{str(row[0]):<35} | ${str(row[1]):<14} | {row[2]}")

    print("\n--- Hunting the Outliers (Top 5 Largest Raw Receipts) ---")
    # Query 2: Finding out exactly which SROIE images tricked the AI!
    cursor.execute('''
        SELECT image_name, merchant, total_amount, category
        FROM receipts
        ORDER BY total_amount DESC
        LIMIT 5
    ''')
    
    outliers = cursor.fetchall()
    print(f"{'IMAGE':<18} | {'MERCHANT':<20} | {'AMOUNT':<12} | {'CATEGORY'}")
    print("-" * 75)
    for row in outliers:
        print(f"{str(row[0]):<18} | {str(row[1])[:20]:<20} | ${str(row[2]):<11} | {row[3]}")

if __name__ == "__main__":
    db_connection = create_database()
    load_data_to_db(db_connection)
    run_sample_query(db_connection)
    db_connection.close()