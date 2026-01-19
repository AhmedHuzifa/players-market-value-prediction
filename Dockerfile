FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/ 




WORKDIR /app

ENV PATH="/app/.venv/bin:${PATH}"


COPY .python-version pyproject.toml uv.lock ./

RUN uv sync --locked

COPY predict.py model.bin ./

EXPOSE 8080

#ENTRYPOINT ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8080"]
