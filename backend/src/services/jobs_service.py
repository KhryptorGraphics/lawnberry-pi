import asyncio
import inspect
from datetime import UTC, datetime, timedelta

from ..core.observability import observability
from ..models.job import Job, JobPriority, JobStatus, JobType

logger = observability.get_logger(__name__)


class JobsService:
    """Job scheduling and execution service."""

    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.job_counter = 0
        self.scheduler_running = False
        self._scheduler_task: asyncio.Task | None = None

    def create_job(
        self,
        name: str,
        job_type: JobType = JobType.SCHEDULED_MOW,
        zones: list[str] = None,
        priority: JobPriority = JobPriority.NORMAL,
        **kwargs,
    ) -> Job:
        """Create a new job."""
        self.job_counter += 1
        job_id = f"job-{self.job_counter:04d}"

        job = Job(
            id=job_id, name=name, job_type=job_type, zones=zones or [], priority=priority, **kwargs
        )

        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Job | None:
        """Get job by ID."""
        return self.jobs.get(job_id)

    def list_jobs(
        self, status: JobStatus | None = None, job_type: JobType | None = None
    ) -> list[Job]:
        """List jobs with optional filtering."""
        jobs = list(self.jobs.values())

        if status:
            jobs = [job for job in jobs if job.status == status]

        if job_type:
            jobs = [job for job in jobs if job.job_type == job_type]

        # Sort by priority (high to low) then by created_at
        jobs.sort(key=lambda j: (-j.priority.value, j.created_at))
        return jobs

    def update_job(self, job_id: str, **updates) -> Job | None:
        """Update job properties."""
        job = self.jobs.get(job_id)
        if not job:
            return None

        for key, value in updates.items():
            if hasattr(job, key):
                setattr(job, key, value)

        return job

    def delete_job(self, job_id: str) -> bool:
        """Delete a job."""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            if job.status == JobStatus.RUNNING:
                # Cancel running job first
                self.cancel_job(job_id)
            del self.jobs[job_id]
            return True
        return False

    def start_job(self, job_id: str) -> bool:
        """Start executing a job."""
        job = self.jobs.get(job_id)
        if not job or job.status != JobStatus.PENDING:
            return False

        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(UTC)

        # Dispatch execution to the navigation/mission pipeline.
        asyncio.create_task(self._execute_job(job))
        return True

    def pause_job(self, job_id: str) -> bool:
        """Pause a running job."""
        job = self.jobs.get(job_id)
        if not job or job.status != JobStatus.RUNNING:
            return False

        job.status = JobStatus.PAUSED
        return True

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job."""
        job = self.jobs.get(job_id)
        if not job or job.status != JobStatus.PAUSED:
            return False

        job.status = JobStatus.RUNNING
        return True

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        job = self.jobs.get(job_id)
        if not job or job.status in [JobStatus.COMPLETED, JobStatus.CANCELLED]:
            return False

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now(UTC)
        return True

    def get_next_scheduled_jobs(self, limit: int = 10) -> list[Job]:
        """Get next jobs scheduled to run."""
        now = datetime.now(UTC)
        scheduled_jobs = [
            job
            for job in self.jobs.values()
            if job.status == JobStatus.PENDING
            and job.enabled
            and job.scheduled_for
            and job.scheduled_for <= now
        ]

        # Sort by priority then scheduled time
        scheduled_jobs.sort(key=lambda j: (-j.priority.value, j.scheduled_for))
        return scheduled_jobs[:limit]

    async def start_scheduler(self):
        """Start the job scheduler."""
        if self.scheduler_running:
            return

        self.scheduler_running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop_scheduler(self):
        """Stop the job scheduler."""
        self.scheduler_running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None

    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.scheduler_running:
            try:
                # Check for jobs to start
                jobs_to_start = self.get_next_scheduled_jobs(5)
                for job in jobs_to_start:
                    self.start_job(job.id)

                # Update recurring job schedules
                self._update_recurring_schedules()

                # Clean up old completed jobs
                self._cleanup_old_jobs()

                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Scheduler loop error",
                    extra={"error": str(e)},
                    exc_info=True,
                )
                observability.record_error(
                    origin="job_scheduler",
                    message="Scheduler loop error",
                    exception=e,
                    metadata={"context": "_scheduler_loop"},
                )
                await asyncio.sleep(60)

    def _update_recurring_schedules(self):
        """Update next_run times for recurring jobs."""
        now = datetime.now(UTC)

        for job in self.jobs.values():
            if (
                job.schedule
                and job.schedule.enabled
                and job.status == JobStatus.COMPLETED
                and job.enabled
            ):
                # Calculate next run time based on schedule
                next_run = self._calculate_next_run(job, now)
                if next_run:
                    job.next_run = next_run
                    job.scheduled_for = next_run
                    job.status = JobStatus.PENDING

    def _calculate_next_run(self, job: Job, from_time: datetime) -> datetime | None:
        """Calculate next run time for a job."""
        if not job.schedule or not job.schedule.start_time:
            return None

        # Simple daily scheduling (can be enhanced for more complex patterns)
        next_date = from_time.date() + timedelta(days=1)
        next_run = datetime.combine(next_date, job.schedule.start_time)
        next_run = next_run.replace(tzinfo=UTC)

        # Check if this day is allowed
        if job.schedule.days_of_week:
            while next_run.weekday() not in job.schedule.days_of_week:
                next_run += timedelta(days=1)

        return next_run

    def _cleanup_old_jobs(self):
        """Remove old completed jobs."""
        cutoff_date = datetime.now(UTC) - timedelta(days=30)

        jobs_to_remove = [
            job_id
            for job_id, job in self.jobs.items()
            if job.status in [JobStatus.COMPLETED, JobStatus.CANCELLED, JobStatus.FAILED]
            and job.completed_at
            and job.completed_at < cutoff_date
        ]

        for job_id in jobs_to_remove:
            del self.jobs[job_id]

    async def _execute_job(self, job: Job):
        """Execute a job by dispatching it to the navigation/mission pipeline."""
        try:
            from ..models.job import JobProgress

            if not job.progress:
                job.progress = JobProgress()

            if job.job_type in (JobType.SCHEDULED_MOW, JobType.MANUAL_MOW, JobType.MAPPING):
                await self._run_mission_job(job)
            elif job.job_type == JobType.RETURN_HOME:
                await self._run_return_home_job(job)
            else:
                # MAINTENANCE and any other type: nothing to drive on the robot.
                job.progress.percentage_complete = 100.0
                job.execution_logs.append(f"{job.job_type} job acknowledged")

            if job.status == JobStatus.RUNNING:
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now(UTC)
                job.last_run = job.completed_at
                job.result_message = job.result_message or "Job completed successfully"

        except Exception as e:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now(UTC)
            job.error_message = str(e)
            job.execution_logs.append(f"Job failed: {e}")

    async def _run_mission_job(self, job: Job) -> None:
        """Create and run a mowing mission, mapping mission status to job progress."""
        from .mission_service import get_mission_service
        from .navigation_service import NavigationService

        nav = NavigationService.get_instance()
        # Prime the mission-service singleton with a real nav service so the
        # status that execute_mission updates is the one we poll here.
        mission_service = get_mission_service(nav)

        waypoints = self._waypoints_for_job(job)
        mission = await mission_service.create_mission(
            name=job.name or f"job-{job.id}", waypoints=waypoints
        )
        await mission_service.start_mission(mission.id)
        job.execution_logs.append(f"Started mission {mission.id} with {len(waypoints)} waypoint(s)")

        started = datetime.now(UTC)
        timeout = timedelta(minutes=job.timeout_minutes or 120)
        while job.status == JobStatus.RUNNING:
            status = mission_service.mission_statuses.get(mission.id)
            if status is not None:
                job.progress.percentage_complete = float(status.completion_percentage)
                job.progress.runtime_minutes = (datetime.now(UTC) - started).total_seconds() / 60.0
                if status.status == "completed":
                    job.progress.percentage_complete = 100.0
                    job.result_message = "Mowing mission completed"
                    break
                if status.status == "failed":
                    job.status = JobStatus.FAILED
                    job.error_message = "Mission failed"
                    break
                if status.status == "aborted":
                    job.status = JobStatus.CANCELLED
                    break
            if datetime.now(UTC) - started > timeout:
                await mission_service.abort_mission(mission.id)
                job.status = JobStatus.FAILED
                job.error_message = "Mission timed out"
                break
            await asyncio.sleep(0.2)

    async def _run_return_home_job(self, job: Job) -> None:
        """Dispatch a return-to-base request to the navigation service."""
        from .navigation_service import NavigationService

        nav = NavigationService.get_instance()
        handler = getattr(nav, "return_to_base", None) or getattr(nav, "return_home", None)
        if handler is not None:
            result = handler()
            if inspect.isawaitable(result):
                await result
        job.progress.percentage_complete = 100.0
        job.execution_logs.append("Return-to-home dispatched")

    def _waypoints_for_job(self, job: Job) -> list:
        """Best-effort coverage waypoints for the job's zones.

        Reads the persisted map configuration and generates a serpentine path
        over the first boundary zone. Returns an empty list when no geometry is
        available (the mission then completes immediately).
        """
        import json
        import os
        from pathlib import Path

        from ..models.mission import MissionWaypoint
        from ..nav.coverage_planner import plan_coverage

        try:
            config_path = (
                Path(os.getenv("LAWNBERRY_DATA_DIR", "./data")) / "map_configuration_v2.json"
            )
            if not config_path.exists():
                return []
            config = json.loads(config_path.read_text())
            boundary: list[tuple[float, float]] = []
            for zone in config.get("zones") or []:
                if zone.get("zone_type") != "boundary":
                    continue
                coords = (zone.get("geometry") or {}).get("coordinates") or []
                ring = coords[0] if coords and isinstance(coords[0], list) else coords
                boundary = [(float(p[1]), float(p[0])) for p in ring if len(p) >= 2]
                break
            if len(boundary) < 3:
                return []
            path, _, _ = plan_coverage(boundary, spacing_m=0.6)
            return [
                MissionWaypoint(lat=lat, lon=lng, blade_on=True, speed=50)
                for (lat, lng) in path[:500]
            ]
        except Exception:
            return []


# Global instance
jobs_service = JobsService()
