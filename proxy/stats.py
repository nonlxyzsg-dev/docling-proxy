"""PostgreSQL stats: schema, worker, queue, middleware, /_stats/vlm endpoint."""
import os, json, asyncio, time, uuid, logging
from datetime import datetime as dt
from fastapi import FastAPI, Request, Response, APIRouter
from proxy.config import (
    STATS_ENABLED, STATS_DB_DSN, STATS_QUEUE_SIZE,
    STATS_BATCH_SIZE, STATS_FLUSH_INTERVAL_SEC,
    VLM_REQUEST_LOG_FILE,
)

logger = logging.getLogger("docling_proxy")


def _vlm_request_log_path_for(date_str: str) -> str:
    """`vlm_requests_<DATE>.jsonl` рядом с базовым именем VLM_REQUEST_LOG_FILE."""
    base_dir = os.path.dirname(VLM_REQUEST_LOG_FILE) or "."
    base = os.path.basename(VLM_REQUEST_LOG_FILE) or "vlm_requests.jsonl"
    stem, ext = os.path.splitext(base)
    if not ext:
        ext = ".jsonl"
    return os.path.join(base_dir, f"{stem}_{date_str}{ext}")


_STATS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS docling_requests (
    id                      BIGSERIAL PRIMARY KEY,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    request_id              UUID,
    filename                TEXT,
    file_size_bytes         BIGINT,
    file_pages              INTEGER,
    doc_type                TEXT,
    pipeline                TEXT,
    http_status             INTEGER,
    duration_total_ms       INTEGER,
    duration_docling_ms     INTEGER,
    duration_queue_wait_ms  INTEGER,
    timings_json            JSONB,
    error_message           TEXT,
    client_ip               TEXT,
    user_agent              TEXT
);
CREATE INDEX IF NOT EXISTS idx_docling_requests_created_at  ON docling_requests (created_at);
CREATE INDEX IF NOT EXISTS idx_docling_requests_doc_type    ON docling_requests (doc_type);
CREATE INDEX IF NOT EXISTS idx_docling_requests_pipeline    ON docling_requests (pipeline);
CREATE INDEX IF NOT EXISTS idx_docling_requests_http_status ON docling_requests (http_status);
"""


async def _stats_init_conn(conn):
    """Codec для JSONB — чтобы в executemany передавать dict напрямую."""
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )


async def _stats_ensure_schema(pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(_STATS_SCHEMA_SQL)


async def _stats_insert_batch(pool, rows: list) -> None:
    query = """
        INSERT INTO docling_requests (
            created_at, request_id, filename, file_size_bytes, file_pages,
            doc_type, pipeline, http_status, duration_total_ms,
            duration_docling_ms, duration_queue_wait_ms, timings_json,
            error_message, client_ip, user_agent
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
    """
    args = [
        (
            r.get("created_at"),
            r.get("request_id"),
            r.get("filename"),
            r.get("file_size_bytes"),
            r.get("file_pages"),
            r.get("doc_type"),
            r.get("pipeline"),
            r.get("http_status"),
            r.get("duration_total_ms"),
            r.get("duration_docling_ms"),
            r.get("duration_queue_wait_ms"),
            r.get("timings_json"),
            r.get("error_message"),
            r.get("client_ip"),
            r.get("user_agent"),
        )
        for r in rows
    ]
    async with pool.acquire() as conn:
        await conn.executemany(query, args)


async def _stats_worker(app: FastAPI) -> None:
    """Корутина-воркер: вытаскивает записи из очереди и пишет пачками."""
    pool = app.state.stats_pool
    queue: asyncio.Queue = app.state.stats_queue
    metrics = app.state.stats_metrics
    batch: list = []
    last_flush = time.time()
    while True:
        remaining = STATS_FLUSH_INTERVAL_SEC - (time.time() - last_flush)
        try:
            if remaining <= 0:
                raise asyncio.TimeoutError
            item = await asyncio.wait_for(queue.get(), timeout=remaining)
            batch.append(item)
            queue.task_done()
        except asyncio.TimeoutError:
            pass
        should_flush = batch and (
            len(batch) >= STATS_BATCH_SIZE
            or (time.time() - last_flush) >= STATS_FLUSH_INTERVAL_SEC
        )
        if should_flush:
            try:
                await _stats_insert_batch(pool, batch)
                metrics["written"] += len(batch)
            except Exception as e:
                metrics["db_errors"] += 1
                logger.warning(
                    f"Stats: batch insert failed ({e}); dropping {len(batch)} records"
                )
            batch = []
            last_flush = time.time()


async def _stats_metrics_loop(app: FastAPI) -> None:
    while True:
        await asyncio.sleep(60)
        m = app.state.stats_metrics
        q: asyncio.Queue = app.state.stats_queue
        logger.info(
            f"Stats: enqueued={m['enqueued']}, written={m['written']}, "
            f"dropped={m['dropped']}, db_errors={m['db_errors']}, "
            f"queue_size={q.qsize()}"
        )


def _stats_enqueue(app: FastAPI, record: dict) -> None:
    """Non-blocking enqueue. Полная очередь → drop + warning (не часто)."""
    if not STATS_ENABLED or getattr(app.state, "stats_queue", None) is None:
        return
    m = app.state.stats_metrics
    try:
        app.state.stats_queue.put_nowait(record)
        m["enqueued"] += 1
    except asyncio.QueueFull:
        m["dropped"] += 1
        if m["dropped"] == 1 or m["dropped"] % 100 == 0:
            logger.warning(
                f"Stats: queue full, dropping record (total dropped={m['dropped']})"
            )


def _stats_set(request: Request, **fields) -> None:
    """Положить метки в stats-контекст текущего запроса. Безопасно при OFF."""
    state = getattr(request, "state", None)
    bag = getattr(state, "stats", None) if state is not None else None
    if bag is None:
        return
    for k, v in fields.items():
        if v is not None:
            bag[k] = v


async def stats_middleware(request: Request, call_next):
    """HTTP middleware that enqueues request stats. Registered in main.py."""
    if not STATS_ENABLED:
        return await call_next(request)
    is_convert_file = (
        request.method == "POST"
        and request.url.path.rstrip("/").endswith("/convert/file")
    )
    if not is_convert_file:
        return await call_next(request)
    from datetime import timezone
    bag = {
        "created_at": dt.now(timezone.utc),
        "request_id": uuid.uuid4(),
        "client_ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
    }
    request.state.stats = bag
    _t = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        bag["http_status"] = 500
        bag["error_message"] = f"{type(e).__name__}: {e}"
        bag["duration_total_ms"] = int((time.time() - _t) * 1000)
        _stats_enqueue(request.app, bag)
        raise
    bag.setdefault("http_status", response.status_code)
    bag.setdefault("duration_total_ms", int((time.time() - _t) * 1000))
    _stats_enqueue(request.app, bag)
    return response


router = APIRouter()


@router.get("/_stats/vlm")
async def stats_vlm(date: str = ""):
    """Агрегированная сводка по vlm_requests_<DATE>.jsonl за дату."""
    if not date:
        date = dt.utcnow().strftime("%Y-%m-%d")
    try:
        dt.strptime(date, "%Y-%m-%d")
    except ValueError:
        return Response(
            content=json.dumps(
                {"error": "invalid date, expected YYYY-MM-DD"},
                ensure_ascii=False,
            ).encode("utf-8"),
            status_code=400,
            headers={"content-type": "application/json"},
        )

    path = _vlm_request_log_path_for(date)
    if not os.path.isfile(path):
        return {
            "date": date,
            "total": 0,
            "by_profile": {},
            "by_model": {},
            "p50_elapsed_ms": None,
            "p95_elapsed_ms": None,
        }

    LIMIT = 100_000
    rows: list = []
    too_large = False
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= LIMIT:
                    too_large = True
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception as e:
        return Response(
            content=json.dumps(
                {"error": f"failed to read log: {e}"}, ensure_ascii=False,
            ).encode("utf-8"),
            status_code=500,
            headers={"content-type": "application/json"},
        )

    if too_large:
        return {
            "date": date,
            "warning": f"file too large (>{LIMIT} lines), use external aggregator",
            "limit": LIMIT,
            "partial_total": len(rows),
        }

    by_profile: dict = {}
    by_model: dict = {}
    elapsed: list = []
    for r in rows:
        prof = r.get("profile") or "unknown"
        st = r.get("status") or "unknown"
        bp = by_profile.setdefault(
            prof, {"ok": 0, "truncated": 0, "error": 0, "truncate_rate": 0.0}
        )
        if st in bp:
            bp[st] += 1
        m = r.get("model")
        if m:
            by_model[m] = by_model.get(m, 0) + 1
        ms = r.get("elapsed_ms")
        if isinstance(ms, (int, float)):
            elapsed.append(ms)

    for prof, counts in by_profile.items():
        total_for_prof = counts["ok"] + counts["truncated"] + counts["error"]
        if total_for_prof > 0:
            counts["truncate_rate"] = round(
                counts["truncated"] / total_for_prof * 100, 2
            )

    elapsed.sort()
    p50 = elapsed[len(elapsed) // 2] if elapsed else None
    p95 = (
        elapsed[min(int(len(elapsed) * 0.95), len(elapsed) - 1)]
        if elapsed else None
    )

    return {
        "date": date,
        "total": len(rows),
        "by_profile": by_profile,
        "by_model": by_model,
        "p50_elapsed_ms": p50,
        "p95_elapsed_ms": p95,
    }
