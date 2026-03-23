FROM ghcr.io/docling-project/docling-serve:main
WORKDIR /proxy

# Локальные пакеты (не зависим от PyPI)
COPY wheels/ /tmp/wheels/
RUN pip install --no-index --find-links=/tmp/wheels/ pymupdf xlrd docxlatex

# Код и конфиг
COPY .env .
COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5003", "--workers", "5"]
