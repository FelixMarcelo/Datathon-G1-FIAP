FROM python:3.10

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-root

COPY app app

RUN tar -xzvf app/model/treino_parte1.tar.gz -C app/model/
RUN tar -xzvf app/model/itens.tar.gz -C app/model

CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
