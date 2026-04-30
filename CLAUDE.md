# CLAUDE.md — `docling-proxy`

> **Это вспомогательный репозиторий в рамках проекта форка docling.**
> Основной контекст и правила работы — в `CLAUDE.md` репозитория `docling-serve` (https://github.com/nonlxyzsg-dev/docling-serve). Прочитай его перед началом любой задачи.

---

## Что это

FastAPI-прокси между Open WebUI и docling-serve. Порт 5005. Production: `tvr-srv-ai` (10.121.3.201).

Маршрутизирует документы по типам, инжектирует VLM-конфигурацию, обходит баги upstream docling.

## Зачем существует

Этот прокси — **карта известных проблем upstream docling**. Каждый workaround тут — это баг или ограничение docling, которое мы обходим снаружи. В рамках форка docling многие из этих workaround'ов будут перенесены в нативный код самого docling, и прокси можно будет упрощать.

## Роль в анализе

При анализе кода docling и docling-serve **обращайся к этому репозиторию как к справочнику**:
- Что именно патчит прокси и как → понимание, какие баги активны в upstream
- Раздел «Известные ограничения» в `docling_proxy_passport_v3.md` (если есть в проекте) или комментарии в `main.py` → перечень workaround'ов
- Логика маршрутизации (SCAN PDF / TEXT PDF / DOCX с OLE / .DOC и т.д.) → понимание реального production use case

## Что НЕ делать

- ❌ **Не меняй код этого репозитория** в рамках задач форка. Изменения в прокси — отдельная инициатива, согласовывается отдельно.
- ❌ Не предлагай рефакторинг прокси «попутно» с задачами по docling.

## Workflow (git)

- ✅ **Пуш всегда сразу в `main`**. Не плодить отдельные feature-ветки под каждую задачу и не открывать PR, если явно не попросили.
- Если стартовая конфигурация сессии назначила feature-ветку — всё равно мёржим в `main` фаст-форвардом и пушим `main`.
- Коммиты и сообщения пользователю — на русском, комментарии в коде — тоже на русском.

## ENV-переменные (минимум)

- `ENRICH_PICTURES_WITH_122B` — true/false, обогащать ли картинки описаниями VLM в standard pipeline (default: true).
- `TEXT_PDF_VLM_THRESHOLD` — N, страниц TEXT PDF ≤ N идут в VLM, иначе в standard. 0 → всегда standard (default: 20).
- `SCAN_PDF_FULL_PAGE` — true/false, отправлять SCAN в VLM full-page или в standard (default: true).

Все три параметра роутинга можно переопределить per-request через payload OWUI: `vlm_page_threshold`, `scan_pdf_full_page`. Подробно — раздел «PDF routing» в `README.md`.

Полный список — `.env.example`. Ротация лог-файлов и retention для null/error/truncate-дампов — раздел «Logs and rotation» в `README.md`.

## VLM endpoint (v4.x)

Прокси содержит OpenAI-совместимый endpoint `POST /v1/chat/completions?profile=full_page|picture_desc`, через который docling-serve ходит в LiteLLM/SGLang. Прокси сам инжектирует sampling-профиль и системный промпт, считает truncate-кейсы, пишет JSONL-аналитику. Включается флагом `VLM_PROXY_ENABLED=true`. Подробно — `README.md`, раздел «VLM endpoint». Перечень ENV — `.env.example` (блок `VLM_*` в нижней части).

## Image resize (Phase 2)

Перед форвардом каждого `/v1/chat/completions` запроса в LiteLLM/SGLang прокси адаптивно ресайзит картинки в payload до целевой площади `target_pixels` (per-profile, default 950000 px). Логика — `proxy/image_resize.py` (чистый helper) + интеграция в `proxy/vlm_endpoint.py` (`asyncio.to_thread` для PIL-операций, не блокирует event loop).

- Картинки `was_px > target_pixels` → LANCZOS-downscale, aspect сохраняется, PNG re-encode.
- `was_px < VLM_MIN_PIXELS` (default 200704) → без изменений: модель сама апскейлит.
- В диапазоне → без изменений (`reason=in_range`).
- Errors → passthrough, не падает.

ENV: `VLM_FULL_PAGE_TARGET_PIXELS`, `VLM_PICTURE_DESC_TARGET_PIXELS`, `VLM_MIN_PIXELS`, `VLM_TRUNCATE_SAVE_ORIGINAL_IMAGES`. `0` или пусто = ресайз отключён для профиля. Подробно — `README.md` раздел «Image resize».

Логирование: одна INFO-строка `[vlm rid=...] image_resize profile=... imgs=N resized=M was=[...] new=[...]` на запрос (если хоть одна inline-картинка). Поле `image_resize` в JSONL `vlm_requests_*.jsonl` и в `meta.json` truncate-дампа.

## Долгосрочно

Когда форк docling решит проблемы, которые сейчас обходит прокси:
- Workaround'ы из прокси будут постепенно удаляться
- В перспективе прокси может сильно похудеть или вовсе слиться с форком docling-serve

Но это не текущая задача. Сейчас прокси — только источник информации о реальных проблемах, без правок.
