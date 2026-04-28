FROM ghcr.io/docling-project/docling-serve:main
WORKDIR /proxy

# stdout/stderr небуферизованные — иначе строки логов могут не появиться
# в `docker compose logs` сразу, что мешает диагностике в реальном времени.
ENV PYTHONUNBUFFERED=1

# Локальные пакеты (не зависим от PyPI)
COPY wheels/ /tmp/wheels/
RUN pip install --no-index --find-links=/tmp/wheels/ pymupdf xlrd docxlatex

# Код и конфиг
COPY .env .
COPY main.py .
# Конфиг логирования uvicorn (timestamps + level + name) — иначе uvicorn
# ставит свои handlers ПОСЛЕ импорта main.py и наш ретрофит из lifespan
# может не сработать.
COPY log_config.json .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5003", "--workers", "1", "--log-config", "/proxy/log_config.json"]
