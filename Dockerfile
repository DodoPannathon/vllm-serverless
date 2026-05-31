FROM runpod/base:0.4.0-cuda12.1.0

WORKDIR /app

COPY requiremenrts.txt .
RUN pip install --no-cache-dir -r requiremenrts.txt

COPY handler.py .

CMD [ "python", "handler.py" ]