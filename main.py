"""docling-proxy entry point: FastAPI app, lifespan wiring, route registration.

All logic lives in proxy.* modules. This file only wires things together.
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from proxy import logging_setup  # noqa: F401  — initializes the docling_proxy logger
from proxy.config import (
    TEXT_PDF_VLM_THRESHOLD, DEFAULT_VLM_SCALE, DEFAULT_IMAGES_SCALE,
    ENRICH_PICTURES_WITH_122B,
    VLM_REQUEST_LOG_RETENTION_DAYS, VLM_REQUEST_LOG_MAX_SIZE_MB,
    NULL_RESPONSE_LOG_RETENTION_DAYS, ERROR_RESPONSE_LOG_RETENTION_DAYS,
    VLM_TRUNCATE_RETENTION_DAYS, VLM_TRUNCATE_LOG_DIR,
    VLM_TRUNCATE_SAVE_PAYLOAD,
    VLM_PROXY_ENABLED, VLM_PROXY_URL, VLM_UPSTREAM_URL,
    VLM_FULL_PAGE_SAMPLING, VLM_PICTURE_DESC_SAMPLING,
    STATS_ENABLED, STATS_DB_DSN, STATS_QUEUE_SIZE,
    STATS_BATCH_SIZE, STATS_FLUSH_INTERVAL_SEC,
)
from proxy.prompts import VLM_FULL_PAGE_PROMPT, VLM_PICTURE_DESC_PROMPT
from proxy.http_client import make_async_client
from proxy.cleanup import (
    cleanup_old_inbox_files, _periodic_inbox_cleanup,
    cleanup_old_vlm_request_logs, cleanup_old_null_response_logs,
    _periodic_logs_cleanup,
    cleanup_old_truncate_dumps, _periodic_truncate_dumps_cleanup,
)
from proxy.stats import (
    _stats_init_conn, _stats_ensure_schema, _stats_worker, _stats_metrics_loop,
    stats_middleware,
    router as stats_router,
)
from proxy.vlm_endpoint import router as vlm_router
from proxy.proxy_handler import router as proxy_router

logger = logging.getLogger("docling_proxy")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ua = logging.getLogger("uvicorn.access")
    logger.info(
        f"[startup-diag] uvicorn.access handlers={len(_ua.handlers)} "
        f"level={logging.getLevelName(_ua.level)} "
        f"propagate={_ua.propagate}"
    )
    app.state.client = make_async_client()
    cleanup_old_inbox_files()
    cleanup_task = asyncio.create_task(_periodic_inbox_cleanup())
    cleanup_old_vlm_request_logs()
    cleanup_old_null_response_logs()
    logs_cleanup_task = asyncio.create_task(_periodic_logs_cleanup())
    cleanup_old_truncate_dumps()
    truncate_cleanup_task = asyncio.create_task(_periodic_truncate_dumps_cleanup())

    logger.info(
        f"[CONFIG] TEXT_PDF_VLM_THRESHOLD={TEXT_PDF_VLM_THRESHOLD}, "
        f"DEFAULT_VLM_SCALE={DEFAULT_VLM_SCALE}, "
        f"DEFAULT_IMAGES_SCALE={DEFAULT_IMAGES_SCALE}"
    )
    logger.info(f"[config] ENRICH_PICTURES_WITH_122B = {ENRICH_PICTURES_WITH_122B}")
    logger.info(
        f"[logs-cleanup] vlm_requests retention={VLM_REQUEST_LOG_RETENTION_DAYS}d"
        f" size_cap={VLM_REQUEST_LOG_MAX_SIZE_MB}MB"
    )
    logger.info(f"[logs-cleanup] null_responses retention={NULL_RESPONSE_LOG_RETENTION_DAYS}d")
    logger.info(f"[logs-cleanup] error_responses retention={ERROR_RESPONSE_LOG_RETENTION_DAYS}d")
    logger.info(f"[logs-cleanup] truncate_dumps retention={VLM_TRUNCATE_RETENTION_DAYS}d")
    logger.info(
        f"[VLM-PROXY] enabled={VLM_PROXY_ENABLED} "
        f"proxy_url={VLM_PROXY_URL or '<unset>'} "
        f"upstream={VLM_UPSTREAM_URL} "
        f"truncate_dir={VLM_TRUNCATE_LOG_DIR} "
        f"save_payload={VLM_TRUNCATE_SAVE_PAYLOAD} "
        f"retention_days={VLM_TRUNCATE_RETENTION_DAYS}"
    )
    logger.info(
        f"[VLM-PROXY] full_page sampling={VLM_FULL_PAGE_SAMPLING} "
        f"prompt_chars={len(VLM_FULL_PAGE_PROMPT)}"
    )
    logger.info(
        f"[VLM-PROXY] picture_desc sampling={VLM_PICTURE_DESC_SAMPLING} "
        f"prompt_chars={len(VLM_PICTURE_DESC_PROMPT)}"
    )

    app.state.stats_pool = None
    app.state.stats_queue = None
    app.state.stats_worker_task = None
    app.state.stats_metrics_task = None
    app.state.stats_metrics = {"enqueued": 0, "written": 0, "dropped": 0, "db_errors": 0}

    if STATS_ENABLED:
        try:
            import asyncpg
            if not STATS_DB_DSN:
                raise RuntimeError("STATS_DB_DSN is empty")
            app.state.stats_pool = await asyncpg.create_pool(
                dsn=STATS_DB_DSN, min_size=1, max_size=5, init=_stats_init_conn
            )
            await _stats_ensure_schema(app.state.stats_pool)
            app.state.stats_queue = asyncio.Queue(maxsize=STATS_QUEUE_SIZE)
            app.state.stats_worker_task = asyncio.create_task(_stats_worker(app))
            app.state.stats_metrics_task = asyncio.create_task(_stats_metrics_loop(app))
            logger.info(
                f"Stats collection enabled (queue={STATS_QUEUE_SIZE}, "
                f"batch={STATS_BATCH_SIZE}, flush={STATS_FLUSH_INTERVAL_SEC}s)"
            )
        except Exception as e:
            logger.error(f"Stats: init failed ({e}); continuing with stats disabled")
            app.state.stats_pool = None
            app.state.stats_queue = None
    else:
        logger.info("Stats collection disabled (STATS_ENABLED=false)")

    yield

    cleanup_task.cancel()
    logs_cleanup_task.cancel()
    truncate_cleanup_task.cancel()

    if app.state.stats_queue is not None:
        try:
            await asyncio.wait_for(app.state.stats_queue.join(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning(
                f"Stats: queue drain timed out; "
                f"{app.state.stats_queue.qsize()} records lost"
            )
    for t in (app.state.stats_worker_task, app.state.stats_metrics_task):
        if t is not None:
            t.cancel()
    if app.state.stats_pool is not None:
        try:
            await app.state.stats_pool.close()
        except Exception as e:
            logger.warning(f"Stats: pool close error: {e}")

    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)
app.middleware("http")(stats_middleware)
app.include_router(stats_router)
app.include_router(vlm_router)
# Catch-all proxy router must be LAST so it does not shadow specific routes.
app.include_router(proxy_router)
