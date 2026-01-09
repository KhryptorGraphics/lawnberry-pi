"""Thor Training Server Uploader Service (BEAD-105).

Handles async upload of recorded training data to the NVIDIA Thor server
for AI model training. Features:
- Async upload over network (HaLow WiFi or standard connection)
- Resumable transfers with chunk-based uploads
- LZ4 compression for bandwidth efficiency
- SHA256 integrity verification
- Upload queue management with persistence
- Automatic retry with exponential backoff

The Thor server is assumed to expose a REST API for receiving training data.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import uuid

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    logger.warning("aiohttp not available - uploads will fail")

try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False
    logger.warning("lz4 not available - compression disabled")


class UploadStatus(str, Enum):
    """Status of an upload task."""
    PENDING = "pending"
    COMPRESSING = "compressing"
    UPLOADING = "uploading"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class UploadTask:
    """Represents a single upload task."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    source_path: str = ""
    file_size: int = 0
    compressed_size: int = 0
    sha256_hash: str = ""

    # Progress tracking
    status: UploadStatus = UploadStatus.PENDING
    bytes_uploaded: int = 0
    chunks_total: int = 0
    chunks_uploaded: int = 0
    progress_percent: float = 0.0

    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Error handling
    retry_count: int = 0
    last_error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d["status"] = self.status.value
        d["created_at"] = self.created_at.isoformat()
        d["started_at"] = self.started_at.isoformat() if self.started_at else None
        d["completed_at"] = self.completed_at.isoformat() if self.completed_at else None
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UploadTask":
        """Create from dictionary."""
        data["status"] = UploadStatus(data["status"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("started_at"):
            data["started_at"] = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            data["completed_at"] = datetime.fromisoformat(data["completed_at"])
        return cls(**data)


@dataclass
class ThorUploaderConfig:
    """Configuration for Thor uploader."""
    # Thor server connection
    thor_base_url: str = "http://thor.local:8000"
    thor_api_key: str = ""

    # Upload settings
    chunk_size: int = 1024 * 1024  # 1MB chunks
    max_retries: int = 5
    retry_delay_base: float = 1.0  # Exponential backoff base
    upload_timeout: float = 300.0  # 5 minutes per chunk

    # Compression
    use_compression: bool = True
    compression_level: int = 3  # LZ4 compression level (1-16)

    # Queue settings
    max_concurrent_uploads: int = 2
    queue_persistence_path: Path = field(default_factory=lambda: Path("data/upload_queue.json"))

    # Cleanup
    delete_after_upload: bool = False
    keep_compressed_cache: bool = False


class ThorUploaderService:
    """Service for uploading training data to Thor server.

    Manages a queue of upload tasks, handles compression, chunked uploads,
    and retry logic for robust data transfer.

    Example usage:
        uploader = ThorUploaderService(config)
        await uploader.initialize()

        task_id = await uploader.queue_upload("/path/to/session.msgpack", "session_123")
        status = await uploader.get_task_status(task_id)

        await uploader.start_processing()
        # ... uploads run in background ...
        await uploader.stop_processing()
    """

    def __init__(self, config: Optional[ThorUploaderConfig] = None):
        """Initialize the uploader service."""
        self.config = config or ThorUploaderConfig()

        # Upload queue
        self._queue: Dict[str, UploadTask] = {}
        self._processing = False
        self._upload_task: Optional[asyncio.Task] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

        # Statistics
        self._total_uploaded_bytes = 0
        self._total_uploads_completed = 0
        self._total_uploads_failed = 0

        # HTTP session
        self._http_session: Optional[aiohttp.ClientSession] = None

        # Callbacks
        self._progress_callbacks: List[Callable[[UploadTask], None]] = []

        self.initialized = False

    async def initialize(self) -> bool:
        """Initialize the uploader service."""
        try:
            # Verify dependencies
            if not HAS_AIOHTTP:
                logger.error("aiohttp required for Thor uploader")
                return False

            # Create compression cache directory
            cache_dir = Path("data/upload_cache")
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Load persistent queue if exists
            await self._load_queue()

            # Create semaphore for concurrent upload limiting
            self._semaphore = asyncio.Semaphore(self.config.max_concurrent_uploads)

            self.initialized = True
            logger.info(f"ThorUploaderService initialized, server: {self.config.thor_base_url}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Thor uploader: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown the uploader service."""
        await self.stop_processing()

        if self._http_session:
            await self._http_session.close()
            self._http_session = None

        # Save queue state
        await self._save_queue()

        self.initialized = False
        logger.info("ThorUploaderService shutdown")

    async def queue_upload(
        self,
        source_path: str,
        session_id: str,
        priority: bool = False,
    ) -> str:
        """Add a file to the upload queue.

        Args:
            source_path: Path to the file to upload.
            session_id: Recording session ID.
            priority: If True, add to front of queue.

        Returns:
            Task ID for tracking the upload.

        Raises:
            FileNotFoundError: If source file doesn't exist.
        """
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        task = UploadTask(
            session_id=session_id,
            source_path=source_path,
            file_size=path.stat().st_size,
        )

        # Calculate hash
        task.sha256_hash = await self._calculate_hash(path)

        self._queue[task.task_id] = task
        await self._save_queue()

        logger.info(f"Queued upload: {session_id} ({task.file_size / 1e6:.1f} MB)")

        return task.task_id

    async def start_processing(self) -> None:
        """Start processing the upload queue."""
        if self._processing:
            return

        self._processing = True
        self._upload_task = asyncio.create_task(self._process_queue())
        logger.info("Started upload queue processing")

    async def stop_processing(self) -> None:
        """Stop processing the upload queue."""
        self._processing = False

        if self._upload_task:
            self._upload_task.cancel()
            try:
                await self._upload_task
            except asyncio.CancelledError:
                pass
            self._upload_task = None

        logger.info("Stopped upload queue processing")

    async def pause_task(self, task_id: str) -> bool:
        """Pause an upload task."""
        if task_id in self._queue:
            task = self._queue[task_id]
            if task.status in (UploadStatus.PENDING, UploadStatus.UPLOADING):
                task.status = UploadStatus.PAUSED
                await self._save_queue()
                return True
        return False

    async def resume_task(self, task_id: str) -> bool:
        """Resume a paused upload task."""
        if task_id in self._queue:
            task = self._queue[task_id]
            if task.status == UploadStatus.PAUSED:
                task.status = UploadStatus.PENDING
                await self._save_queue()
                return True
        return False

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel an upload task."""
        if task_id in self._queue:
            task = self._queue[task_id]
            if task.status not in (UploadStatus.COMPLETED,):
                task.status = UploadStatus.CANCELLED
                await self._save_queue()
                return True
        return False

    async def retry_task(self, task_id: str) -> bool:
        """Retry a failed upload task."""
        if task_id in self._queue:
            task = self._queue[task_id]
            if task.status == UploadStatus.FAILED:
                task.status = UploadStatus.PENDING
                task.retry_count = 0
                task.last_error = ""
                await self._save_queue()
                return True
        return False

    async def get_task_status(self, task_id: str) -> Optional[UploadTask]:
        """Get status of a specific upload task."""
        return self._queue.get(task_id)

    def list_tasks(
        self,
        status: Optional[UploadStatus] = None,
        limit: int = 100,
    ) -> List[UploadTask]:
        """List upload tasks with optional filtering."""
        tasks = list(self._queue.values())

        if status:
            tasks = [t for t in tasks if t.status == status]

        # Sort by created_at descending
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        return tasks[:limit]

    async def _process_queue(self) -> None:
        """Main queue processing loop."""
        logger.info("Upload queue processor started")

        while self._processing:
            try:
                # Get pending tasks
                pending = [
                    t for t in self._queue.values()
                    if t.status == UploadStatus.PENDING
                ]

                if pending:
                    # Process tasks with concurrency limit
                    tasks = []
                    for upload_task in pending[:self.config.max_concurrent_uploads]:
                        tasks.append(self._upload_file(upload_task))

                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)

                # Small delay between queue checks
                await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(5.0)

        logger.info("Upload queue processor stopped")

    async def _upload_file(self, task: UploadTask) -> bool:
        """Upload a single file to Thor server."""
        async with self._semaphore:
            try:
                task.status = UploadStatus.UPLOADING
                task.started_at = datetime.now(timezone.utc)
                self._notify_progress(task)

                source_path = Path(task.source_path)

                # Compress if enabled
                if self.config.use_compression and HAS_LZ4:
                    task.status = UploadStatus.COMPRESSING
                    self._notify_progress(task)

                    compressed_path = await self._compress_file(source_path)
                    upload_path = compressed_path
                    task.compressed_size = compressed_path.stat().st_size
                else:
                    upload_path = source_path
                    task.compressed_size = task.file_size

                # Calculate chunks
                task.chunks_total = (task.compressed_size + self.config.chunk_size - 1) // self.config.chunk_size
                task.status = UploadStatus.UPLOADING
                self._notify_progress(task)

                # Upload chunks
                success = await self._upload_chunks(task, upload_path)

                if success:
                    task.status = UploadStatus.COMPLETED
                    task.completed_at = datetime.now(timezone.utc)
                    task.progress_percent = 100.0
                    self._total_uploaded_bytes += task.compressed_size
                    self._total_uploads_completed += 1

                    # Cleanup
                    if self.config.delete_after_upload:
                        source_path.unlink(missing_ok=True)

                    if not self.config.keep_compressed_cache and upload_path != source_path:
                        upload_path.unlink(missing_ok=True)

                    logger.info(f"Upload completed: {task.session_id}")
                else:
                    raise Exception("Chunk upload failed")

                self._notify_progress(task)
                await self._save_queue()
                return True

            except Exception as e:
                task.retry_count += 1
                task.last_error = str(e)

                if task.retry_count >= self.config.max_retries:
                    task.status = UploadStatus.FAILED
                    self._total_uploads_failed += 1
                    logger.error(f"Upload failed after {task.retry_count} retries: {task.session_id} - {e}")
                else:
                    task.status = UploadStatus.PENDING
                    # Exponential backoff
                    delay = self.config.retry_delay_base * (2 ** task.retry_count)
                    logger.warning(f"Upload retry {task.retry_count}/{self.config.max_retries} in {delay}s: {task.session_id}")
                    await asyncio.sleep(delay)

                self._notify_progress(task)
                await self._save_queue()
                return False

    async def _upload_chunks(self, task: UploadTask, file_path: Path) -> bool:
        """Upload file in chunks with resume support."""
        if self._http_session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.upload_timeout)
            self._http_session = aiohttp.ClientSession(timeout=timeout)

        headers = {
            "X-Session-ID": task.session_id,
            "X-Task-ID": task.task_id,
            "X-SHA256": task.sha256_hash,
            "X-Total-Chunks": str(task.chunks_total),
            "X-Total-Size": str(task.compressed_size),
            "X-Compressed": str(self.config.use_compression and HAS_LZ4),
        }

        if self.config.thor_api_key:
            headers["Authorization"] = f"Bearer {self.config.thor_api_key}"

        with open(file_path, 'rb') as f:
            # Skip already uploaded chunks (resume support)
            f.seek(task.chunks_uploaded * self.config.chunk_size)

            for chunk_num in range(task.chunks_uploaded, task.chunks_total):
                if not self._processing:
                    return False

                chunk_data = f.read(self.config.chunk_size)
                if not chunk_data:
                    break

                chunk_headers = {
                    **headers,
                    "X-Chunk-Index": str(chunk_num),
                    "X-Chunk-Size": str(len(chunk_data)),
                }

                try:
                    upload_url = f"{self.config.thor_base_url}/api/upload/chunk"
                    async with self._http_session.put(
                        upload_url,
                        data=chunk_data,
                        headers=chunk_headers,
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(f"Upload failed: {response.status} - {error_text}")

                        task.chunks_uploaded = chunk_num + 1
                        task.bytes_uploaded = task.chunks_uploaded * self.config.chunk_size
                        task.progress_percent = (task.chunks_uploaded / task.chunks_total) * 100
                        self._notify_progress(task)

                except aiohttp.ClientError as e:
                    raise Exception(f"Network error: {e}")

        # Finalize upload
        try:
            finalize_url = f"{self.config.thor_base_url}/api/upload/finalize"
            async with self._http_session.post(
                finalize_url,
                headers=headers,
                json={"task_id": task.task_id, "session_id": task.session_id},
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Finalize failed: {response.status} - {error_text}")

        except aiohttp.ClientError as e:
            raise Exception(f"Finalize network error: {e}")

        return True

    async def _compress_file(self, source_path: Path) -> Path:
        """Compress file using LZ4."""
        if not HAS_LZ4:
            return source_path

        cache_dir = Path("data/upload_cache")
        compressed_path = cache_dir / f"{source_path.stem}.lz4"

        # Run compression in thread pool to avoid blocking
        def compress():
            with open(source_path, 'rb') as src:
                with lz4.frame.open(compressed_path, 'wb', compression_level=self.config.compression_level) as dst:
                    while chunk := src.read(self.config.chunk_size):
                        dst.write(chunk)

        await asyncio.to_thread(compress)

        original_size = source_path.stat().st_size
        compressed_size = compressed_path.stat().st_size
        ratio = compressed_size / original_size * 100

        logger.info(f"Compressed {source_path.name}: {original_size/1e6:.1f}MB -> {compressed_size/1e6:.1f}MB ({ratio:.1f}%)")

        return compressed_path

    async def _calculate_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        def hash_file():
            h = hashlib.sha256()
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    h.update(chunk)
            return h.hexdigest()

        return await asyncio.to_thread(hash_file)

    async def _load_queue(self) -> None:
        """Load queue from persistent storage."""
        try:
            if self.config.queue_persistence_path.exists():
                with open(self.config.queue_persistence_path, 'r') as f:
                    data = json.load(f)

                for task_data in data.get("tasks", []):
                    task = UploadTask.from_dict(task_data)
                    # Reset in-progress tasks to pending
                    if task.status in (UploadStatus.UPLOADING, UploadStatus.COMPRESSING):
                        task.status = UploadStatus.PENDING
                    self._queue[task.task_id] = task

                logger.info(f"Loaded {len(self._queue)} tasks from persistent queue")

        except Exception as e:
            logger.warning(f"Failed to load upload queue: {e}")

    async def _save_queue(self) -> None:
        """Save queue to persistent storage."""
        try:
            self.config.queue_persistence_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "tasks": [t.to_dict() for t in self._queue.values()],
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }

            with open(self.config.queue_persistence_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save upload queue: {e}")

    def _notify_progress(self, task: UploadTask) -> None:
        """Notify progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(task)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def add_progress_callback(self, callback: Callable[[UploadTask], None]) -> None:
        """Add a callback for upload progress notifications."""
        self._progress_callbacks.append(callback)

    def remove_progress_callback(self, callback: Callable[[UploadTask], None]) -> None:
        """Remove a progress callback."""
        if callback in self._progress_callbacks:
            self._progress_callbacks.remove(callback)

    async def health_check(self) -> Dict[str, Any]:
        """Return health status for /health endpoint."""
        pending_count = len([t for t in self._queue.values() if t.status == UploadStatus.PENDING])
        uploading_count = len([t for t in self._queue.values() if t.status == UploadStatus.UPLOADING])
        failed_count = len([t for t in self._queue.values() if t.status == UploadStatus.FAILED])

        return {
            "service": "thor_uploader",
            "initialized": self.initialized,
            "processing": self._processing,
            "thor_url": self.config.thor_base_url,
            "queue_size": len(self._queue),
            "pending_uploads": pending_count,
            "active_uploads": uploading_count,
            "failed_uploads": failed_count,
            "total_uploaded_bytes": self._total_uploaded_bytes,
            "total_uploads_completed": self._total_uploads_completed,
            "total_uploads_failed": self._total_uploads_failed,
            "compression_enabled": self.config.use_compression and HAS_LZ4,
            "aiohttp_available": HAS_AIOHTTP,
            "lz4_available": HAS_LZ4,
        }

    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to Thor server."""
        try:
            if self._http_session is None:
                self._http_session = aiohttp.ClientSession()

            headers = {}
            if self.config.thor_api_key:
                headers["Authorization"] = f"Bearer {self.config.thor_api_key}"

            start = time.perf_counter()
            async with self._http_session.get(
                f"{self.config.thor_base_url}/health",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10.0),
            ) as response:
                latency_ms = (time.perf_counter() - start) * 1000
                return {
                    "connected": response.status == 200,
                    "status_code": response.status,
                    "latency_ms": round(latency_ms, 2),
                    "server_url": self.config.thor_base_url,
                }

        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "server_url": self.config.thor_base_url,
            }


# Singleton instance management
_uploader_instance: Optional[ThorUploaderService] = None


def get_thor_uploader() -> ThorUploaderService:
    """Get or create the ThorUploaderService singleton."""
    global _uploader_instance
    if _uploader_instance is None:
        _uploader_instance = ThorUploaderService()
    return _uploader_instance


def set_thor_uploader(uploader: ThorUploaderService) -> None:
    """Set the ThorUploaderService singleton (for dependency injection)."""
    global _uploader_instance
    _uploader_instance = uploader


__all__ = [
    "ThorUploaderService",
    "ThorUploaderConfig",
    "UploadTask",
    "UploadStatus",
    "get_thor_uploader",
    "set_thor_uploader",
]
