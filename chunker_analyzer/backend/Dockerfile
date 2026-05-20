FROM python:3.12-slim

WORKDIR /app

# Kopiujemy pliki projektu (pyproject + ewentualnie lock)
COPY pyproject.toml ./
# jeśli masz lockfile, odkomentuj:
# COPY poetry.lock ./

# Instalacja zależności
RUN pip install --no-cache-dir .

COPY app.py .

ENV PORT=8000
ENV DATA_ROOT=/data

EXPOSE 8000

CMD ["sh", "-c", "python app.py"]