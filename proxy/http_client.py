"""HTTP client factory + concurrency semaphore."""
import asyncio, os, httpx, logging

logger = logging.getLogger("docling_proxy")


def make_async_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(1200.0, connect=10.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        follow_redirects=False,
    )


_semaphore = None
_semaphore_value = 0

def get_semaphore(max_docs: int) -> asyncio.Semaphore:
    global _semaphore, _semaphore_value
    if _semaphore is None or _semaphore_value != max_docs:
        _semaphore = asyncio.Semaphore(max_docs)
        _semaphore_value = max_docs
        # Лог для диагностики: при workers>1 каждый воркер создаст свой
        # семафор и мы увидим в логах разные pid'ы — это и есть симптом
        # рассинхрона. При workers=1 pid должен быть один и тот же.
        logger.info(f"[semaphore] (re)created: max_docs={max_docs}  pid={os.getpid()}")
    return _semaphore
