"""Адаптивный контроль исходящих VLM-запросов (capacity gate).

Gateway придерживает форвард в LiteLLM/vLLM ровно настолько, насколько занята
модель. Сигнал ёмкости — Prometheus /metrics инстанса vLLM:
- kv_cache_usage_perc  — главный лимитер для запросов переменного размера;
- num_requests_waiting — backlog (>0 = модель уже копит очередь);
- num_requests_running — текущий батч (для самокалибровки KV/запрос).

Принцип (НЕ семафор-константа): пока есть запас (waiting <= W И eff_kv < порога)
— форвардим немедленно. Иначе — ждём внутри proxy и отпускаем, как только запас
вернулся. При исчерпании бюджета ожидания — last-resort форвард (страницу не
теряем). При недоступных метриках — fallback на локальный кап.

Анти-овершут: /metrics кэшируется (опрос ~300 мс), между опросами держим
счётчик допущенных (_admitted_since_poll) и оценку KV/запрос (kv/running),
чтобы пачка одновременных запросов не проскочила разом по устаревшему чтению.

Состояние — в памяти процесса. Корректно при UVICORN_WORKERS=1 (см. Dockerfile).
"""
import time
import asyncio
import logging
from contextlib import asynccontextmanager

import httpx

from proxy.config import (
    VLM_GATE_ENABLED, VLM_METRICS_URL,
    VLM_METRIC_RUNNING, VLM_METRIC_WAITING, VLM_METRIC_KV,
    VLM_METRICS_POLL_MS, VLM_METRICS_TIMEOUT_MS, VLM_METRICS_STALE_MS,
    VLM_GATE_KV_THRESHOLD, VLM_GATE_WAITING_MAX, VLM_GATE_DEFAULT_PER_REQ_KV,
    VLM_GATE_FALLBACK_MAX_INFLIGHT, VLM_GATE_WAIT_BUDGET_SEC,
    VLM_GATE_LAST_RESORT_FRACTION, VLM_GATE_RECHECK_MS,
)

logger = logging.getLogger("docling_proxy")


def _parse_metric(text: str, name: str) -> list[float]:
    """Достать значения серии Prometheus по точному имени метрики.

    Учитывает границу имени: `vllm:num_requests_waiting` НЕ матчит
    `vllm:num_requests_waiting_by_reason{...}` (после имени должен идти
    '{' или пробел/таб).
    """
    out: list[float] = []
    nlen = len(name)
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] == "#":
            continue
        if not line.startswith(name):
            continue
        sep = line[nlen:nlen + 1]
        if sep not in ("{", " ", "\t"):
            continue
        try:
            out.append(float(line.rsplit(None, 1)[1]))
        except (ValueError, IndexError):
            continue
    return out


def _extract(text: str) -> tuple[int, int, float]:
    """(running, waiting, kv) из текста /metrics. Counts — сумма серий, kv — max."""
    run = _parse_metric(text, VLM_METRIC_RUNNING)
    wait = _parse_metric(text, VLM_METRIC_WAITING)
    kv = _parse_metric(text, VLM_METRIC_KV)
    return (
        int(sum(run)) if run else 0,
        int(sum(wait)) if wait else 0,
        max(kv) if kv else 0.0,
    )


class VlmCapacityGate:
    """Адаптивный допуск исходящих VLM-запросов по реальной ёмкости модели."""

    def __init__(self) -> None:
        self._cond = asyncio.Condition()
        # snapshot: ts — момент последнего УСПЕШНОГО чтения (для staleness).
        self._snap = {"running": 0, "waiting": 0, "kv": 0.0, "ts": 0.0, "ok": False}
        self._in_flight = 0
        self._admitted_since_poll = 0
        self._last_resort_count = 0
        self._poll_task: asyncio.Task | None = None
        self._stop = False
        self._client: httpx.AsyncClient | None = None
        self._fail_logged = False

        # Параметры (в секундах/долях).
        self._poll_interval = max(VLM_METRICS_POLL_MS, 50) / 1000.0
        self._fetch_timeout = max(VLM_METRICS_TIMEOUT_MS, 100) / 1000.0
        self._stale_sec = max(VLM_METRICS_STALE_MS, 200) / 1000.0
        self._kv_threshold = VLM_GATE_KV_THRESHOLD
        self._waiting_max = VLM_GATE_WAITING_MAX
        self._default_per_req_kv = max(VLM_GATE_DEFAULT_PER_REQ_KV, 0.0001)
        self._fallback_max = max(VLM_GATE_FALLBACK_MAX_INFLIGHT, 1)
        self._last_resort_after = VLM_GATE_WAIT_BUDGET_SEC * VLM_GATE_LAST_RESORT_FRACTION
        self._recheck = max(VLM_GATE_RECHECK_MS, 20) / 1000.0

    # ── lifecycle ──
    def start(self, client: httpx.AsyncClient) -> None:
        if not VLM_GATE_ENABLED:
            logger.info("[vlm-gate] disabled (VLM_GATE_ENABLED=false) — passthrough")
            return
        self._client = client
        self._stop = False
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info(
            f"[vlm-gate] enabled url={VLM_METRICS_URL} poll={self._poll_interval*1000:.0f}ms "
            f"kv_threshold={self._kv_threshold} waiting_max={self._waiting_max} "
            f"last_resort_after={self._last_resort_after:.0f}s "
            f"fallback_max={self._fallback_max}"
        )

    async def stop(self) -> None:
        self._stop = True
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):
                pass
            self._poll_task = None

    # ── poller ──
    async def _poll_loop(self) -> None:
        assert self._client is not None
        while not self._stop:
            try:
                r = await self._client.get(VLM_METRICS_URL, timeout=self._fetch_timeout)
                if r.status_code == 200:
                    run, wait, kv = _extract(r.text)
                    async with self._cond:
                        self._snap = {
                            "running": run, "waiting": wait, "kv": kv,
                            "ts": time.monotonic(), "ok": True,
                        }
                        self._admitted_since_poll = 0
                        self._cond.notify_all()
                    if self._fail_logged:
                        logger.info("[vlm-gate] /metrics recovered")
                        self._fail_logged = False
                else:
                    self._mark_fail(f"HTTP {r.status_code}")
            except Exception as e:
                self._mark_fail(f"{type(e).__name__}: {e}")
            await asyncio.sleep(self._poll_interval)

    def _mark_fail(self, reason: str) -> None:
        # ts НЕ обновляем — возраст снапшота растёт → переход в fallback по staleness.
        self._snap["ok"] = False
        if not self._fail_logged:
            logger.warning(f"[vlm-gate] /metrics unavailable ({reason}) — fallback mode")
            self._fail_logged = True

    # ── admission ──
    def _capacity_locked(self) -> tuple[bool, str]:
        """Есть ли ёмкость прямо сейчас. Вызывать под self._cond."""
        snap = self._snap
        age = time.monotonic() - snap["ts"]
        if age > self._stale_sec:
            # Метрики устарели/недоступны — локальный кап.
            if self._in_flight < self._fallback_max:
                return True, "fallback"
            return False, "fallback_full"
        if snap["waiting"] > self._waiting_max:
            return False, "waiting"
        if snap["running"] > 0 and snap["kv"] > 0:
            per_req_kv = snap["kv"] / snap["running"]
        else:
            per_req_kv = self._default_per_req_kv
        eff_kv = snap["kv"] + self._admitted_since_poll * per_req_kv
        if eff_kv >= self._kv_threshold:
            return False, "kv"
        return True, "ok"

    async def _acquire(self, rid8: str) -> bool:
        """Дождаться ёмкости и занять слот. Возвращает last_resort (bool)."""
        t0 = time.monotonic()
        deadline = t0 + self._last_resort_after
        async with self._cond:
            while True:
                now = time.monotonic()
                ok, reason = self._capacity_locked()
                if ok:
                    self._in_flight += 1
                    self._admitted_since_poll += 1
                    waited = now - t0
                    if waited > 0.05:
                        logger.info(
                            f"[vlm-gate rid={rid8}] admitted after {waited*1000:.0f}ms "
                            f"({reason}) in_flight={self._in_flight} "
                            f"snap_wait={self._snap['waiting']} snap_kv={self._snap['kv']:.2f}"
                        )
                    return False
                if now >= deadline:
                    self._in_flight += 1
                    self._admitted_since_poll += 1
                    self._last_resort_count += 1
                    logger.warning(
                        f"[vlm-gate rid={rid8}] LAST-RESORT admit after "
                        f"{(now-t0)*1000:.0f}ms (reason={reason}); "
                        f"snap run={self._snap['running']} wait={self._snap['waiting']} "
                        f"kv={self._snap['kv']:.2f} in_flight={self._in_flight} "
                        f"— OVERLOAD signal (total last_resort={self._last_resort_count})"
                    )
                    return True
                timeout = min(self._recheck, deadline - now)
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    pass

    async def _release(self) -> None:
        async with self._cond:
            if self._in_flight > 0:
                self._in_flight -= 1
            self._cond.notify(1)

    @asynccontextmanager
    async def admit(self, rid8: str):
        """Контекст-менеджер вокруг исходящего форварда. yield last_resort:bool."""
        if not VLM_GATE_ENABLED:
            yield False
            return
        last_resort = await self._acquire(rid8)
        try:
            yield last_resort
        finally:
            await self._release()

    def snapshot(self) -> dict:
        """Текущее состояние гейта — для /_gate и диагностики."""
        snap = self._snap
        age_ms = int((time.monotonic() - snap["ts"]) * 1000) if snap["ts"] else None
        return {
            "enabled": VLM_GATE_ENABLED,
            "in_flight": self._in_flight,
            "admitted_since_poll": self._admitted_since_poll,
            "last_resort_total": self._last_resort_count,
            "metrics": {
                "running": snap["running"],
                "waiting": snap["waiting"],
                "kv_cache_usage_perc": snap["kv"],
                "ok": snap["ok"],
                "age_ms": age_ms,
                "stale": (age_ms is None) or (age_ms > self._stale_sec * 1000),
            },
            "thresholds": {
                "kv_threshold": self._kv_threshold,
                "waiting_max": self._waiting_max,
                "fallback_max_inflight": self._fallback_max,
                "last_resort_after_sec": self._last_resort_after,
            },
        }


# Синглтон процесса (workers=1).
gate = VlmCapacityGate()
