# Статистика обработки запросов в PostgreSQL

Опциональная подсистема `docling-proxy`, собирающая метаданные о каждом
успешном и неуспешном запросе `POST /v1/convert/file` в отдельную таблицу
`docling_requests`. По умолчанию **выключена** — включается одним env-флагом.

Гарантия: при `STATS_ENABLED=false` накладных расходов ноль — очередь,
пул и воркер не создаются. При `STATS_ENABLED=true` запись идёт
fire-and-forget из фоновой корутины батчами, горячий путь запроса никогда
не ждёт БД. Падение PostgreSQL не ломает обработку запросов.

## Как включить

### 1. Добавить в `.env`

```env
STATS_ENABLED=true
STATS_DB_DSN=postgresql://docling_stats:PASSWORD@10.121.3.201:5433/docling_stats
# необязательные — значения по умолчанию:
STATS_QUEUE_SIZE=10000
STATS_BATCH_SIZE=50
STATS_FLUSH_INTERVAL_SEC=5
```

DSN в логи никогда не попадает. При старте прокси видно только:
`Stats collection enabled (queue=..., batch=..., flush=...s)`
или, если инициализация провалилась:
`Stats: init failed (...); continuing with stats disabled`.

### 2. Подготовить PostgreSQL

Рекомендуется **отдельная БД `docling_stats`** на том же PG-сервере
(10.121.3.201:5433). Обоснование:

- логическая изоляция от продовых БД (OWUI, n8n) — бэкапы и retention
  независимые;
- отдельный пользователь/роль с правами только на одну таблицу, меньше
  риск при компрометации DSN;
- таблица метаданных растёт заметно, её удобно чистить партиционированием
  отдельно от бизнес-данных.

```sql
-- под суперпользователем, один раз:
CREATE USER docling_stats WITH PASSWORD 'strong-password';
CREATE DATABASE docling_stats OWNER docling_stats;
GRANT ALL PRIVILEGES ON DATABASE docling_stats TO docling_stats;
```

Схему таблицы и индексы прокси создаёт **сам при старте** (idempotent
`CREATE TABLE IF NOT EXISTS` + `CREATE INDEX IF NOT EXISTS`), миграции
руками гонять не нужно.

### 3. Положить `asyncpg` в `wheels/`

Сборка Docker-образа идёт с `--no-index` (без PyPI). Перед билдом нужно
положить wheel в каталог `wheels/`:

```bash
pip download --no-deps -d wheels/ asyncpg
```

И добавить в `Dockerfile` строку `asyncpg` к списку пакетов:

```dockerfile
RUN pip install --no-index --find-links=/tmp/wheels/ pymupdf xlrd docxlatex asyncpg
```

`asyncpg` импортируется лениво внутри lifespan: если его нет, но
`STATS_ENABLED=false`, прокси поднимается штатно. Если
`STATS_ENABLED=true` и `asyncpg` не установлен — в лог уходит
`Stats: init failed (No module named 'asyncpg'); continuing with stats disabled`,
прокси продолжает работать без статистики.

### 4. Рестарт контейнера

```bash
docker compose up -d --force-recreate docling-proxy
```

## Как отключить

В `.env` поставить `STATS_ENABLED=false` (или удалить строку) и
рестартовать контейнер. Очередь при штатном shutdown дописывается в БД
в пределах 10 секунд; всё, что не успело — теряется, лог сообщает
`Stats: queue drain timed out; N records lost`.

## Что собирается

Таблица `docling_requests`:

| Поле                     | Тип          | Смысл                                              |
|--------------------------|--------------|----------------------------------------------------|
| `id`                     | bigserial    | PK                                                 |
| `created_at`             | timestamptz  | Момент прихода запроса в прокси (UTC)              |
| `request_id`             | uuid         | Идентификатор для корреляции с логами              |
| `filename`               | text         | Имя исходного файла                                |
| `file_size_bytes`        | bigint       | Размер загруженного файла                          |
| `file_pages`             | integer      | Страниц в PDF (null для не-PDF)                    |
| `doc_type`               | text         | `SCAN` / `TEXT_SHORT` / `TEXT_LONG` / `DOC` / `DOC_CONFLUENCE` / `XLS` / `DOCX_OLE` / `OTHER` / `UNSUPPORTED` |
| `pipeline`               | text         | `vlm` / `standard` / `ocr-sdk` / `gotenberg` / `xls_native` / `confluence_html` |
| `http_status`            | integer      | Итоговый HTTP-статус ответа клиенту                |
| `duration_total_ms`      | integer      | Общее время прокси (конец - начало запроса)        |
| `duration_docling_ms`    | integer      | Время ожидания ответа от docling-serve              |
| `duration_queue_wait_ms` | integer      | Время ожидания в семафоре concurrency              |
| `timings_json`           | jsonb        | Словарь `timings` из ответа docling-serve           |
| `error_message`          | text         | Тело ответа (500 байт) при не-200                   |
| `client_ip`              | text         | IP клиента (OWUI/n8n)                              |
| `user_agent`             | text         | `User-Agent` клиента                               |

Индексы: `created_at`, `doc_type`, `pipeline`, `http_status`.

**Никаких содержимых документов** (ни исходника, ни markdown-ответа)
в таблице нет — только метаданные. Имя файла хранится как есть.

## Метрики самой статистики

Раз в минуту в INFO-лог пишется:

```
Stats: enqueued=N, written=M, dropped=K, db_errors=E, queue_size=Q
```

- `enqueued` — добавлено в очередь
- `written` — записано в БД
- `dropped` — отброшено из-за переполнения очереди
- `db_errors` — неудачных батч-вставок
- `queue_size` — текущий размер очереди

Если `queue_size` стабильно растёт и `dropped > 0` — БД не справляется,
увеличьте `STATS_BATCH_SIZE` или разбирайтесь с производительностью PG.

## Примеры SQL-запросов

Топ-10 самых медленных запросов за последние сутки:

```sql
SELECT created_at, filename, doc_type, pipeline,
       file_pages, duration_total_ms, http_status
FROM docling_requests
WHERE created_at > NOW() - INTERVAL '1 day'
ORDER BY duration_total_ms DESC NULLS LAST
LIMIT 10;
```

Среднее время обработки по типам документов за неделю:

```sql
SELECT doc_type, pipeline,
       COUNT(*)                                  AS n,
       ROUND(AVG(duration_total_ms))             AS avg_ms,
       ROUND(percentile_cont(0.5)
             WITHIN GROUP (ORDER BY duration_total_ms)) AS p50_ms,
       ROUND(percentile_cont(0.95)
             WITHIN GROUP (ORDER BY duration_total_ms)) AS p95_ms
FROM docling_requests
WHERE created_at > NOW() - INTERVAL '7 days'
  AND http_status = 200
GROUP BY doc_type, pipeline
ORDER BY n DESC;
```

Количество ошибок за сутки:

```sql
SELECT http_status, COUNT(*) AS n,
       array_agg(DISTINCT left(error_message, 80)) AS sample_errors
FROM docling_requests
WHERE created_at > NOW() - INTERVAL '1 day'
  AND http_status >= 400
GROUP BY http_status
ORDER BY n DESC;
```

Какая доля времени тратится внутри docling vs в прокси:

```sql
SELECT doc_type, pipeline,
       ROUND(AVG(duration_docling_ms))   AS avg_docling_ms,
       ROUND(AVG(duration_total_ms))     AS avg_total_ms,
       ROUND(AVG(duration_total_ms - COALESCE(duration_docling_ms, 0))) AS avg_proxy_ms
FROM docling_requests
WHERE created_at > NOW() - INTERVAL '1 day'
  AND http_status = 200
GROUP BY doc_type, pipeline
ORDER BY avg_total_ms DESC;
```

Timings docling по шагам (только для записей, где есть `timings_json`):

```sql
SELECT doc_type,
       ROUND(AVG((timings_json->'page_parse'->>'total_seconds')::numeric), 2) AS page_parse_s,
       ROUND(AVG((timings_json->'doc_build'->>'total_seconds')::numeric), 2)  AS doc_build_s,
       ROUND(AVG((timings_json->'doc_enrich'->>'total_seconds')::numeric), 2) AS doc_enrich_s
FROM docling_requests
WHERE timings_json IS NOT NULL
  AND created_at > NOW() - INTERVAL '1 day'
GROUP BY doc_type;
```

(Названия ключей внутри `timings_json` — как их отдаёт docling-serve,
при необходимости подгоняйте под фактический формат ответа.)

## Безопасность

- В БД пишутся только метаданные, без содержимого документа.
- DSN читается из env и нигде в логах не появляется.
- Ошибки записи в БД никогда не вызывают HTTP-ошибку у клиента.
- При переполнении очереди новые записи отбрасываются (не теряется
  обработка запросов, теряется только телеметрия).
