"""
app/core/backup.py

Automated backup utility for RAG database assets (SQLite + FAISS indices)
incorporating timestamped archive generation and a strict retention policy.
"""

import os
import shutil
import time
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)


def backup_data_assets(backup_dir: str = "data/backups", max_backups: int = 5) -> str | None:
    """
    Bundles the SQLite database file and the FAISS vector index folder into a
    single timestamped zip archive. Retains only the last 'max_backups' archives.
    Returns the file path of the created archive if successful, else None.
    """
    try:
        # Check if there is anything to backup first
        db_exists = os.path.exists(settings.SQLITE_DB_PATH)
        # For FAISS, check if index directory exists and contains files
        faiss_exists = os.path.exists(settings.FAISS_INDEX_PATH) and len(os.listdir(settings.FAISS_INDEX_PATH)) > 0

        if not db_exists and not faiss_exists:
            logger.info("Backup skipped: no database or vector index assets found on disk.")
            return None

        os.makedirs(backup_dir, exist_ok=True)
        # Use millisecond precision to prevent filename collisions in rapid tests
        timestamp = int(time.time() * 1000)
        archive_name = f"backup_{timestamp}"
        
        # Temp bundle directory to copy files before zipping
        bundle_dir = os.path.join(backup_dir, f"bundle_{timestamp}")
        os.makedirs(bundle_dir, exist_ok=True)

        # 1. Copy SQLite database if it exists
        if db_exists:
            shutil.copy2(settings.SQLITE_DB_PATH, os.path.join(bundle_dir, "db.sqlite3"))

        # 2. Copy FAISS index if it exists
        if faiss_exists:
            shutil.copytree(settings.FAISS_INDEX_PATH, os.path.join(bundle_dir, "faiss_index"))

        # Zip bundle directory
        archive_dest = os.path.join(backup_dir, archive_name)
        shutil.make_archive(archive_dest, "zip", bundle_dir)
        
        # Clean up bundle temp folder
        shutil.rmtree(bundle_dir)
        
        backup_file = f"{archive_dest}.zip"
        logger.info("Backup successfully generated: %s", backup_file)

        # 3. Apply retention limits (purge oldest zip files exceeding limit)
        backups = sorted(
            [os.path.join(backup_dir, f) for f in os.listdir(backup_dir) 
             if f.startswith("backup_") and f.endswith(".zip")],
            key=os.path.getmtime
        )
        while len(backups) > max_backups:
            oldest = backups.pop(0)
            try:
                os.remove(oldest)
                logger.info("Pruned old backup per retention policy: %s", oldest)
            except Exception as exc:
                logger.error("Failed to delete expired backup file %s: %s", oldest, exc)

        return backup_file
    except Exception as e:
        logger.error("Failed to create automated assets backup: %s", e, exc_info=True)
        return None
