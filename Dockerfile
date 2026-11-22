# Build: docker build -t mech-bot .   (match runs it with no network, read-only root, 1 CPU, 512 MB, map at /arena)
FROM python:3.13-slim
# 1 CPU quota: extra BLAS/OpenMP threads burn it and get the process throttled past the 50 ms deadline
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN pip install --no-cache-dir numpy==2.2.6
WORKDIR /bot
COPY bot/ .
ENTRYPOINT ["python3", "-u", "/bot/main.py"]
