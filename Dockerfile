# Базовый образ docling-serve, запинен по digest для воспроизводимости
# и защиты от silent breaking changes в upstream.
# Чтобы обновить — выполнить:
#   docker pull ghcr.io/docling-project/docling-serve:main
#   docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/docling-project/docling-serve:main
# и обновить хэш ниже + проверить acceptance-тесты.
FROM ghcr.io/docling-project/docling-serve@sha256:00135f1e84a925d898de02ea493f7582175d959c560bf0af27d48ad9f199c8dd
WORKDIR /proxy

# stdout/stderr небуферизованные — иначе строки логов могут не появиться
# в `docker compose logs` сразу, что мешает диагностике в реальном времени.
ENV PYTHONUNBUFFERED=1

# Системные бинарники для rar НЕ ставим: базовый образ — CentOS Stream 9,
# и в нём уже есть /usr/bin/bsdtar (libarchive 3.5.x читает RAR4/RAR5).
# rarfile подхватывает bsdtar автоматически. 7z — через py7zr (чистый Python),
# zip/tar — stdlib. Если в будущем понадобится системный пакет — это dnf, не apt.

# Локальные пакеты (не зависим от PyPI).
# py7zr и rarfile нужны для .7z и .rar. Их (и транзитивные зависимости py7zr:
# pycryptodomex, pyzstd, pyppmd, pybcj, multivolumefile, inflate64, brotli,
# texttable) необходимо предварительно положить в wheels/. Если каких-то
# wheel'ов нет — уберите py7zr/rarfile из строки: zip/tar продолжат работать.
COPY wheels/ /tmp/wheels/
RUN pip install --no-index --find-links=/tmp/wheels/ pymupdf xlrd docxlatex py7zr rarfile

# Код и конфиг
COPY .env .
COPY main.py .
# Пакет proxy/ — без него main.py падает на `from proxy.config import ...`.
# WORKDIR=/proxy, импорт идёт как `proxy.*`, поэтому кладём в /proxy/proxy/.
COPY proxy/ ./proxy/
# Конфиг логирования uvicorn (timestamps + level + name) — иначе uvicorn
# ставит свои handlers ПОСЛЕ импорта main.py и наш ретрофит из lifespan
# может не сработать.
COPY log_config.json .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5003", "--workers", "1", "--log-config", "/proxy/log_config.json"]
