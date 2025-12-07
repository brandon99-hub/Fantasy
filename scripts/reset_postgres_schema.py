"""
Script to reset and reinitialize PostgreSQL schema with corrected data types
"""
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from backend.src.core.postgres_db import PostgresManagerDB
from backend.src.core.config import get_settings
import psycopg

settings = get_settings()

def reset_schema():
    """Drop and recreate FPL data tables only (preserves manager tables)"""
    print("🔄 Resetting PostgreSQL FPL data tables...")
    print("⚠️  Manager tables will be preserved")
    
    db = PostgresManagerDB()
    
    # Test connection first
    if not db.test_connection():
        print("❌ Cannot connect to database")
        return False
    
    print("✅ Connected to database")
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                print("📋 Dropping existing FPL data tables...")
                
                # Drop ONLY FPL data tables in reverse order of dependencies
                # DO NOT drop manager tables (fpl_managers, manager_gameweeks, etc.)
                tables_to_drop = [
                    'predictions',
                    'model_metrics',
                    'fixture_odds',
                    'player_history',
                    'fixtures',
                    'gameweeks',
                    'players',
                    'teams',
                    'data_updates'
                ]
                
                for table in tables_to_drop:
                    try:
                        cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
                        print(f"  ✓ Dropped {table}")
                    except Exception as e:
                        print(f"  ⚠ Could not drop {table}: {e}")
                
                conn.commit()
                print("✅ FPL data tables dropped (manager tables preserved)")
    except Exception as e:
        print(f"❌ Error dropping tables: {e}")
        return False
    
    # Reinitialize schema (will create FPL tables and ensure manager tables exist)
    print("\n🔨 Creating FPL data tables with corrected data types...")
    if db.initialize_schema():
        print("✅ Schema reinitialized successfully!")
        print("\n📊 Updated schema includes:")
        print("  • Corrected INTEGER/BIGINT types for players table")
        print("  • Corrected INTEGER types for gameweeks table")
        print("  • New data_updates tracking table")
        print("  • All manager tables preserved with existing data")
        return True
    else:
        print("❌ Schema initialization failed")
        return False

if __name__ == "__main__":
    success = reset_schema()
    if success:
        print("\n✅ Database is ready for data loading!")
        print("   Run: python scripts/start_all.py")
    else:
        print("\n❌ Schema reset failed")
        sys.exit(1)
