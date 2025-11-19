# Old Files Backup

This folder contains files from the old project structure before reorganization.

## ⚠️ Important

**DO NOT USE THESE FILES** - They are kept only for reference.

The active codebase is now in:
- `backend/src/` - All Python code
- `frontend/` - All frontend code

## Contents

### Old Core Files
- `main.py` - Old 744-line monolithic API (now split into modules in `backend/src/api/routes/`)
- `database.py` - Old database file (now `backend/src/core/database.py`)
- `optimizer.py` - Old optimizer (now `backend/src/core/optimizer.py`)
- `data_collector.py` - Old data collector (now `backend/src/core/data_collector.py`)
- `fpl_api_client.py` - Old API client (now `backend/src/services/fpl_api_client.py`)

### Old Folders
- `models/` - Old models folder (now `backend/src/models/`)
- `utils/` - Old utils folder (now `backend/src/utils/`)

### Old Database
- `fpl_data.db` - Old database file (now `backend/data/fpl_data.db`)

### Old Test/Demo Files
- `check_db.py`
- `demo_improvements.py`
- `test_imports.py`
- `test_improvements.py`
- `validate_checklist.py`

### Old Setup Files
- `quick_setup.bat`
- `setup_environment.py`

### Old Assets
- `attached_assets/` - Old attached files

## Can I Delete This?

**Yes**, after verifying the new structure works correctly:

1. Test the new structure:
   ```bash
   python -m backend.src.api.main
   cd frontend && npm run dev
   ```

2. If everything works, you can safely delete this folder:
   ```bash
   # Windows PowerShell
   Remove-Item -Recurse -Force old_files_backup/
   
   # Linux/Mac
   rm -rf old_files_backup/
   ```

## Reorganization Date

**November 19, 2025**

See `MIGRATION_GUIDE.md` and `REORGANIZATION_SUMMARY.md` in the root directory for details on what changed.

