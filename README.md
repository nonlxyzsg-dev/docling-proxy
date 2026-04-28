# docling-proxy

FastAPI-прокси между Open WebUI и docling-serve. Маршрутизирует документы по типам, инжектирует VLM-конфигурацию, обходит баги upstream docling. Production: `tvr-srv-ai` (10.121.3.201:5005).

См. также:
- [`CLAUDE.md`](CLAUDE.md) — назначение, workflow, отношение к форку docling.
- [`README-STATS.md`](README-STATS.md) — опциональная PostgreSQL-статистика запросов.
- [`.env.example`](.env.example) — все ENV-переменные с пояснениями.

---

## VLM endpoint

С v4.x прокси умеет работать как **VLM gateway** между docling-serve и LiteLLM/SGLang. Endpoint `POST /v1/chat/completions?profile=full_page|picture_desc` принимает OpenAI-совместимый chat completions request, инжектирует sampling-профиль и системный промпт из ENV (если клиент их не задал), форвардит на `VLM_UPSTREAM_URL`, считает счётчики и при `finish_reason=length` сохраняет дамп для анализа.

### Зачем

После калибровки sampling Qwen3.5-122B-GPTQ-Int4 (апрель 2026) единственный надёжный способ убрать лупы — стабильно подставлять профиль `temperature=1.0, top_p=0.8, top_k=40, min_p=0.1, presence_penalty=2.0`. Прокси берёт это на себя: docling-serve ничего про sampling не знает, всё в ENV прокси.

### Архитектура

```
OWUI → docling-proxy (5005) → docling-serve (5002) → docling-proxy (5005, /v1/chat/completions)
                                                                           │
                                                                           ▼
                                                       LiteLLM (4000) → SGLang (9989)
```

`build_vlm_pipeline_model_api()` и `build_picture_description_api()` в `main.py` подставляют в конфиг docling-serve `url=${VLM_PROXY_URL}?profile=...` — но только при `VLM_PROXY_ENABLED=true`. Иначе работают по-старому, напрямую в LiteLLM.

### Включение

1. **Сначала деплой с `VLM_PROXY_ENABLED=false`** — endpoint работает (доступен для тестов), но docling-serve в него не ходит. Это безопасный старт.
2. Smoke-тест endpoint'а (см. ниже).
3. Поменять в `.env`: `VLM_PROXY_ENABLED=true`. Рестарт: `docker compose up -d`.
4. Если что-то пошло не так — вернуть `false` и рестартануть. Без перебилда.

### Профили

| Профиль | Когда применяется | Дефолт sampling |
|---|---|---|
| `full_page` | full-page OCR страницы (через `vlm_pipeline_model_api`) | `temperature=1.0, top_p=0.8, top_k=40, min_p=0.1, presence_penalty=2.0, max_tokens=3072` |
| `picture_desc` | описание вырезанной картинки (через `picture_description_api`) | `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=1024` |

Любой параметр можно отключить, оставив пустым в `.env` — тогда upstream получит default из SGLang. Клиентский override (тело запроса) всегда приоритетнее ENV.

### Промпты

Дефолтные промпты зашиты в `main.py` (`DEFAULT_VLM_FULL_PAGE_PROMPT`, `DEFAULT_VLM_PICTURE_DESC_PROMPT`). Чтобы переопределить без пересборки — создать файл и указать путь в `VLM_PROMPT_FULL_PAGE_FILE` / `VLM_PROMPT_PICTURE_DESC_FILE`. Каталог `/proxy/prompts/` уже примонтирован через `docker-compose.yml` (`/docker-shared/docling-proxy-prompts`).

### Аутентификация

Прокси не проверяет входящий `Authorization` (стоит в корпсети). При форварде в upstream подменяет заголовок на `Bearer ${VLM_UPSTREAM_API_KEY}`. Клиент (docling-serve) может присылать любую строку, например `Bearer cant-be-empty` — это нормально.

### Аналитика

- `${VLM_REQUEST_LOG_FILE}` — JSONL append-only с дневной ротацией: фактически создаются файлы `vlm_requests_<YYYY-MM-DD>.jsonl`. По строке на запрос: `request_id`, `profile`, `model`, `prompt_tokens`, `completion_tokens`, `finish_reason`, `elapsed_ms`, `status` (`ok`/`truncated`/`error`).
- `${VLM_TRUNCATE_LOG_DIR}/<DATE>/<request_id>/` — детализация для `finish_reason=length`: `request.json` (без base64), `response.json`, `meta.json`, `page_*.png`. При `VLM_TRUNCATE_SAVE_PAYLOAD=false` — только `meta.json`.
- Cleanup-задача в `lifespan` раз в сутки удаляет каталоги старше `VLM_TRUNCATE_RETENTION_DAYS` (по умолчанию 30).
- `GET /_stats/vlm?date=YYYY-MM-DD` — агрегированная сводка. Без даты — текущий день. Лимит 100k строк, дальше — рекомендация перейти на внешний агрегатор.

### Streaming

Streaming (`stream: true`) сейчас не поддерживается — endpoint вернёт 400 `streaming is not supported by the VLM proxy`. Текущая инфраструктура streaming не использует.

---

## Smoke-тесты

### Простой ответ

```bash
curl -sS -X POST 'http://10.121.3.201:5005/v1/chat/completions?profile=full_page' \
  -H 'Authorization: Bearer cant-be-empty' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.5-122B-agent-50k",
    "messages": [{"role": "user", "content": "Скажи hello world"}]
  }'
```

Ожидаемое: 200 OK, корректный chat-completion ответ. В `vlm_requests_*.jsonl` появилась строка `status=ok profile=full_page`. В response — заголовок `X-Request-ID`.

### Truncate-кейс

```bash
curl -sS -X POST 'http://10.121.3.201:5005/v1/chat/completions?profile=full_page' \
  -H 'Authorization: Bearer cant-be-empty' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.5-122B-agent-50k",
    "messages": [{"role": "user", "content": "Перечисли числа от 1 до 1000 через запятую"}],
    "max_tokens": 50
  }'
```

Ответ обрывается, `finish_reason=length`. На диске:

- `${VLM_TRUNCATE_LOG_DIR}/<сегодня>/<request_id>/meta.json` существует
- `request.json`, `response.json` сохранены
- `page.png` нет (картинки в запросе не было)

### Сводка

```bash
curl -sS http://10.121.3.201:5005/_stats/vlm | jq
curl -sS http://10.121.3.201:5005/_stats/vlm?date=2026-04-28 | jq
```

---

## Logs and rotation

Прокси пишет четыре независимых потока логов на диск. У каждого свой retention в днях, у `vlm_requests_*.jsonl` дополнительно есть размерный cap. Чистка — раз в сутки в `lifespan` (плюс холостой проход при старте контейнера). Само логирование — append-only из hot path; ничего не блокируется.

| Лог | Куда | Retention (ENV) | Размерный cap | Период чистки |
|---|---|---|---|---|
| `vlm_requests_<DATE>.jsonl` | каталог `${VLM_REQUEST_LOG_FILE}` | `VLM_REQUEST_LOG_RETENTION_DAYS=90` | `VLM_REQUEST_LOG_MAX_SIZE_MB=5120` (суммарно) | сутки |
| `null_response_*.json` | `LOG_DIR` (`./logs`) | `NULL_RESPONSE_LOG_RETENTION_DAYS=30` | — | сутки |
| `error_response_*.json` | `LOG_DIR` (`./logs`) | `ERROR_RESPONSE_LOG_RETENTION_DAYS=30` | — | сутки |
| `truncated/<DATE>/<request_id>/` | `${VLM_TRUNCATE_LOG_DIR}` | `VLM_TRUNCATE_RETENTION_DAYS=30` | — | сутки |

При старте в логах появляются строки:

```
[logs-cleanup] vlm_requests retention=90d size_cap=5120MB
[logs-cleanup] null_responses retention=30d
[logs-cleanup] error_responses retention=30d
[logs-cleanup] truncate_dumps retention=30d
```

Размерный cap — страховка от переполнения диска, если поток запросов внезапно вырос. Удаление по cap'у логируется WARNING:

```
[logs-cleanup] vlm_requests size-cap: removed vlm_requests_2026-01-15.jsonl (614400 bytes, mtime=...)
[logs-cleanup] vlm_requests size-cap: removed 2 file(s), freed 1228 MB (was 5320 MB, cap 5120 MB)
```

Чистка не трогает файлы с другим именем — ровно `vlm_requests_*.jsonl`, `null_response_*.json`, `error_response_*.json`. Каталоги truncate-дампов — по mtime директории.

---

## Известные параметры (краткий перечень)

Все ENV — в `.env.example`. Ключевое:

| ENV | Назначение |
|---|---|
| `ENRICH_PICTURES_WITH_122B` | true/false, обогащать ли картинки описаниями VLM в standard pipeline (default: `true`). |
| `VLM_PROXY_ENABLED` | Главный флаг. `false` → docling-serve ходит мимо прокси (старое поведение). `true` → через прокси. |
| `VLM_PROXY_URL` | Self-reference, инжектируется в конфиги docling-serve. |
| `VLM_UPSTREAM_URL` / `VLM_UPSTREAM_API_KEY` | Куда форвардит прокси. Если пусто — берётся `DEFAULT_VLM_URL`/`DEFAULT_VLM_API_KEY`. |
| `VLM_FULL_PAGE_*` | Sampling-профиль для full-page OCR. |
| `VLM_PICTURE_DESC_*` | Sampling-профиль для описания картинок. |
| `VLM_PROMPT_*_FILE` | Кастомные файлы промптов (опционально). |
| `VLM_REQUEST_LOG_FILE` | JSONL счётчиков. |
| `VLM_TRUNCATE_LOG_DIR` | Каталог детализации truncate-кейсов. |
| `VLM_TRUNCATE_SAVE_PAYLOAD` | `true`: с request/response/картинками; `false`: только meta. |
| `VLM_TRUNCATE_RETENTION_DAYS` | Сколько дней хранить дампы. |
