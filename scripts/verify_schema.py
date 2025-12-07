"""
Verify the PostgreSQL schema has the correct data types
"""
import sys
from pathlib import Path

backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from backend.src.core.postgres_db import PostgresManagerDB
import psycopg

def check_schema():
    """Check the actual data types in the database"""
    print("Checking PostgreSQL schema...")
    
    db = PostgresManagerDB()
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                # Check players table columns
                cur.execute("""
                    SELECT column_name, data_type, character_maximum_length
                    FROM information_schema.columns
                    WHERE table_name = 'players'
                    AND column_name IN ('now_cost', 'total_points', 'minutes', 'transfers_in', 'transfers_out')
                    ORDER BY column_name;
                """)
                
                print("\nPlayers table column types:")
                for row in cur.fetchall():
                    print(f"  {row['column_name']}: {row['data_type']}")
                
                # Check gameweeks table columns
                cur.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = 'gameweeks'
                    AND column_name IN ('average_entry_score', 'highest_score', 'most_selected', 'most_captained')
                    ORDER BY column_name;
                """)
                
                print("\nGameweeks table column types:")
                for row in cur.fetchall():
                    print(f"  {row['column_name']}: {row['data_type']}")
                
                # Check if data_updates table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'data_updates'
                    );
                """)
                
                exists = cur.fetchone()['exists']
                print(f"\ndata_updates table exists: {exists}")
                
                return True
    except Exception as e:
        print(f"Error checking schema: {e}")
        return False

if __name__ == "__main__":
    check_schema()
