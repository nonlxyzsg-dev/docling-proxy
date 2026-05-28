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

`build_vlm_pipeline_model_api()` и `build_picture_description_api()` в `proxy/builders.py` подставляют в конфиг docling-serve `url=${VLM_PROXY_URL}?profile=...` — но только при `VLM_PROXY_ENABLED=true`. Иначе работают по-старому, напрямую в LiteLLM.

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

Дефолтные промпты в `proxy/prompts.py` (`DEFAULT_VLM_FULL_PAGE_PROMPT`, `DEFAULT_VLM_PICTURE_DESC_PROMPT`). Чтобы переопределить без пересборки — создать файл и указать путь в `VLM_PROMPT_FULL_PAGE_FILE` / `VLM_PROMPT_PICTURE_DESC_FILE`.

### Аутентификация

Прокси не проверяет входящий `Authorization` (стоит в корпсети). При форварде в upstream подменяет заголовок на `Bearer ${VLM_UPSTREAM_API_KEY}`.

### Аналитика

- `${VLM_REQUEST_LOG_FILE}` — JSONL append-only с дневной ротацией: фактически создаются файлы `vlm_requests_<YYYY-MM-DD>.jsonl`. По строке на запрос: `request_id`, `profile`, `model`, `prompt_tokens`, `completion_tokens`, `finish_reason`, `elapsed_ms`, `status` (`ok`/`truncated`/`error`), и опционально `image_resize` (см. ниже).
- `${VLM_TRUNCATE_LOG_DIR}/<DATE>/<request_id>/` — детализация для `finish_reason=length`: `request.json` (без base64), `response.json`, `meta.json`, `page_*.png`. При `VLM_TRUNCATE_SAVE_PAYLOAD=false` — только `meta.json`.
- `GET /_stats/vlm?date=YYYY-MM-DD` — агрегированная сводка.

### Streaming

Streaming (`stream: true`) сейчас не поддерживается — endpoint вернёт 400.

---

## Image resize (Phase 2)

Перед форвардом каждого `/v1/chat/completions` запроса в LiteLLM/SGLang прокси адаптивно ресайзит каждую картинку из `messages[].content[].image_url.url` (если это `data:image/png;base64,...` или JPEG) к целевой площади `target_pixels`. Ресайз в worker thread через `asyncio.to_thread`, не блокирует event loop.

Зачем: Qwen3.5-VL имеет жёсткие пороги `min_pixels=200 704` и `max_pixels=1 003 520`. Всё что выше — модель ресайзит сама, неуправляемо, теряя детали хаотично. Эмпирический оптимум — 950 000 px (отчёт «Устранение лупов» 25.04.2026, раздел 5).

### Поведение per размер

| Исходный размер | Действие |
|---|---|
| `was_px > target_pixels` | LANCZOS-downscale, `new_w*new_h ≈ target_pixels`, aspect сохраняется. PNG re-encode. |
| `min_pixels ≤ was_px ≤ target_pixels` | Без изменений (`reason=in_range`). |
| `was_px < min_pixels` | Без изменений (`reason=too_small`) — модель сама апскейлит до своих минимумов. |
| Не PNG/JPEG | Без изменений (`reason=unsupported_mime`). |
| Ошибка decode/encode | Без изменений (`reason=error`). Не падает. |

### Per-profile конфиг

| ENV | Дефолт | Эффект |
|---|---|---|
| `VLM_FULL_PAGE_TARGET_PIXELS` | 950000 | Целевая площадь для `?profile=full_page`. 0 = ресайз отключён. |
| `VLM_PICTURE_DESC_TARGET_PIXELS` | 950000 | Целевая площадь для `?profile=picture_desc`. 0 = отключён. |
| `VLM_MIN_PIXELS` | 200704 | Картинки ниже не трогаем — модель апскейлит сама. |
| `VLM_TRUNCATE_SAVE_ORIGINAL_IMAGES` | false | При truncate-дампе сохранять и оригинал (до ресайза) как `page_<i>_original.png` рядом с фактически отправленной `page_<i>.png`. |

### Размеры до/после (для понимания)

| Что | До (vlm_scale=2.0) | После (target=950k) |
|---|---|---|
| A4 full_page | 1190×1684 = 2.00 МПкс | ≈815×1156 = 942k |
| A3 full_page | 1684×2382 = 4.01 МПкс | ≈580×820 = 475k → wait, sqrt(0.95/4.01)*size → ≈819×1158 ≈ 949k |
| A0 full_page | 4768×6740 = 32.1 МПкс | ≈821×1161 ≈ 953k |
| Picture crop 200×200 px = 40k | ниже min | без изменений (40k) |
| Picture crop 600×600 px = 360k | в range | без изменений (360k) |

### Логирование

При наличии хотя бы одной inline-картинки в запросе пишется одна INFO-строка:

```
[vlm rid=abc12345] image_resize profile=full_page imgs=2 resized=2 was=[2073600,2002048] new=[948000,948320] elapsed=152ms
```

Если `imgs=0` (картинок не было) — строка не пишется. Если `resized=0` — строка пишется с массивом `was=` для прозрачности.

В `vlm_requests_<DATE>.jsonl` к каждой записи (когда `has_image=true`) добавляется поле:

```json
"image_resize": {"imgs": 2, "resized": 2, "avg_was_px": 2002048, "avg_new_px": 948160, "elapsed_ms": 152, "target_pixels": 950000}
```

В truncate-дампе `meta.json` — то же поле (если был ресайз). В каталоге дампа `page_<i>.png` — фактически отправленные (после ресайза) картинки. Если `VLM_TRUNCATE_SAVE_ORIGINAL_IMAGES=true` — рядом сохраняются `page_<i>_original.png`.

### Производительность

На CPU средний overhead — 100–250 мс на картинку A4 (2 МПкс). На A0 (32 МПкс) — до 1 сек. Это меньше чем VLM-инференс на странице (5–30 сек), некритично. Ресайз идёт в worker thread, event loop не блокируется.

### Отключение

`VLM_FULL_PAGE_TARGET_PIXELS=0` или `VLM_PICTURE_DESC_TARGET_PIXELS=0` — passthrough для соответствующего профиля. В стартовых логах `[VLM-PROXY] ... target_pixels=disabled`.

---

## PDF routing

Прокси автоматически выбирает pipeline для каждого PDF между `vlm` (full-page OCR через Qwen3.5-122B) и `standard` (текстовое извлечение + picture description). Решение зависит от типа документа (SCAN / TEXT) и числа страниц.

| Случай | Параметр | Действие |
|---|---|---|
| SCAN PDF | `scan_pdf_full_page=true` | → `vlm` (full-page) |
| SCAN PDF | `scan_pdf_full_page=false` | → `standard` (docling сам OCR'ит) |
| TEXT PDF, страниц `≤ vlm_page_threshold` | `vlm_page_threshold=N` | → `vlm` |
| TEXT PDF, страниц `> vlm_page_threshold` | | → `standard` |
| TEXT PDF, любой объём | `vlm_page_threshold=0` | → `standard` (никогда не VLM) |

**Приоритет источников:** `payload (form-data)` > `.env` > хардкод. Хардкоды: `vlm_page_threshold=20`, `scan_pdf_full_page=true`.

Ручной override `pipeline=vlm|standard` в payload OWUI по-прежнему перебивает любой автоматический роутинг.

---

## Архивы

Если в `/v1/convert/file` приходит архив, прокси разворачивает его, обрабатывает каждый вложенный документ **так же, как обычную загрузку в чат** (XLS/DOC/PDF-scan/DOCX-OLE/standard/vlm — весь обычный роутинг), и склеивает результат в один markdown с заголовками-разделителями по файлам.

Распаковка **рекурсивная**: архив внутри архива тоже разворачивается (до `ARCHIVE_MAX_DEPTH` уровней).

**Поддерживаемые форматы:**

| Формат | Зависимость |
|---|---|
| `.zip` | stdlib |
| `.tar`, `.tar.gz`/`.tgz`, `.tar.bz2`, `.tar.xz` | stdlib |
| одиночные `.gz` / `.bz2` / `.xz` | stdlib |
| `.7z` | `py7zr` (wheels) |
| `.rar` | `rarfile` (wheels) + бинарник `unrar`/`bsdtar` (см. Dockerfile) |

**Поведение:**

- Неподдерживаемые/битые файлы внутри архива — **пропускаются с пометкой** в итоговом markdown (раздел «Примечания»), запрос не падает.
- Если `py7zr`/`rarfile` или системный бинарник недоступны — соответствующий архив помечается как необработанный, остальное содержимое обрабатывается.
- Служебный мусор архиваторов (`__MACOSX/`, `.DS_Store`, `Thumbs.db`, AppleDouble `._*`) отбрасывается.
- Ответ — стандартный docling-формат: `document.md_content` со склеенным markdown + блок `proxy_diagnostics` (`files_total`, `files_ok`, `files_skipped`, `notes`).

**Защита от zip-бомб** (ENV, приоритет: `.env` > хардкод):

| Параметр | Хардкод | Смысл |
|---|---|---|
| `ARCHIVE_PROCESSING_ENABLED` | `true` | глобальный вкл/выкл фичи |
| `ARCHIVE_MAX_DEPTH` | `5` | макс. глубина вложенности (архив в архиве) |
| `ARCHIVE_MAX_FILES` | `200` | макс. число извлекаемых файлов суммарно |
| `ARCHIVE_MAX_TOTAL_MB` | `500` | макс. суммарный распакованный объём |

При срабатывании лимита остаток содержимого пропускается с пометкой в «Примечаниях».

---

## Smoke-тесты

### Простой ответ

```bash
curl -sS -X POST 'http://10.121.3.201:5005/v1/chat/completions?profile=full_page' \
  -H 'Authorization: Bearer cant-be-empty' \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-122B-agent-50k","messages":[{"role":"user","content":"Скажи hello world"}]}'
```

Ожидаемое: 200 OK, корректный chat-completion ответ. В `vlm_requests_*.jsonl` — строка `status=ok profile=full_page`. В response — заголовок `X-Request-ID`.

### Запрос с большой картинкой (проверка ресайза)

```bash
curl -sS -X POST 'http://10.121.3.201:5005/v1/chat/completions?profile=full_page' \
  -H 'Content-Type: application/json' \
  -d @big_image_request.json
```

В логах: `[vlm rid=...] image_resize profile=full_page imgs=1 resized=1 was=[~2M] new=[~948k]`.

### Запрос с маленькой картинкой (не ресайзится)

```bash
curl -sS -X POST 'http://10.121.3.201:5005/v1/chat/completions?profile=picture_desc' \
  -H 'Content-Type: application/json' \
  -d @small_image_request.json
```

В логах: `[vlm rid=...] image_resize profile=picture_desc imgs=1 resized=0 was=[40000] new=[40000]`.

### Сводка

```bash
curl -sS http://10.121.3.201:5005/_stats/vlm | jq
```

---

## Logs and rotation

Прокси пишет четыре независимых потока логов на диск. У каждого свой retention в днях, у `vlm_requests_*.jsonl` дополнительно есть размерный cap. Чистка — раз в сутки в `lifespan` (плюс холостой проход при старте контейнера).

| Лог | Куда | Retention (ENV) | Размерный cap | Период чистки |
|---|---|---|---|---|
| `vlm_requests_<DATE>.jsonl` | каталог `${VLM_REQUEST_LOG_FILE}` | `VLM_REQUEST_LOG_RETENTION_DAYS=90` | `VLM_REQUEST_LOG_MAX_SIZE_MB=5120` | сутки |
| `null_response_*.json` | `LOG_DIR` (`./logs`) | `NULL_RESPONSE_LOG_RETENTION_DAYS=30` | — | сутки |
| `error_response_*.json` | `LOG_DIR` (`./logs`) | `ERROR_RESPONSE_LOG_RETENTION_DAYS=30` | — | сутки |
| `truncated/<DATE>/<request_id>/` | `${VLM_TRUNCATE_LOG_DIR}` | `VLM_TRUNCATE_RETENTION_DAYS=30` | — | сутки |

---

## Известные параметры (краткий перечень)

Все ENV — в `.env.example`. Ключевое:

| ENV | Назначение |
|---|---|
| `ENRICH_PICTURES_WITH_122B` | true/false, обогащать ли картинки описаниями VLM в standard pipeline (default: `true`). |
| `VLM_PROXY_ENABLED` | Главный флаг. `false` → docling-serve ходит мимо прокси. `true` → через прокси. |
| `VLM_PROXY_URL` | Self-reference, инжектируется в конфиги docling-serve. |
| `VLM_UPSTREAM_URL` / `VLM_UPSTREAM_API_KEY` | Куда форвардит прокси. |
| `VLM_FULL_PAGE_*` / `VLM_PICTURE_DESC_*` | Sampling-профили per profile. |
| `VLM_FULL_PAGE_TARGET_PIXELS` / `VLM_PICTURE_DESC_TARGET_PIXELS` | Image resize target по профилям. 0=disabled. |
| `VLM_MIN_PIXELS` | Картинки ниже не ресайзятся. |
| `VLM_TRUNCATE_SAVE_ORIGINAL_IMAGES` | В truncate-дамп сохранять и оригинал картинки (до ресайза). |
| `VLM_PROMPT_*_FILE` | Кастомные файлы промптов (опционально). |
| `VLM_REQUEST_LOG_FILE` / `VLM_TRUNCATE_LOG_DIR` | Пути логов и truncate-дампов. |
| `VLM_TRUNCATE_SAVE_PAYLOAD` | `true`: с request/response/картинками; `false`: только meta. |
| `VLM_TRUNCATE_RETENTION_DAYS` | Сколько дней хранить дампы. |
