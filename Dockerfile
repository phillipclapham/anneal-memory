FROM python:3.12-slim

RUN pip install --no-cache-dir anneal-memory "mcp-proxy>=0.11.0,<1.0"

RUN useradd --create-home anneal \
    && mkdir -p /data \
    && chown anneal:anneal /data

USER anneal

VOLUME /data

EXPOSE 8080

ENTRYPOINT ["mcp-proxy", "--host=0.0.0.0", "--port=8080", "--", "anneal-memory", "--db", "/data/memory.db"]
