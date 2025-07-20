FROM python:3.9-slim
CMD ["python", "--version"]

WORKDIR /workspace

COPY requirements.txt .
COPY . .

RUN apt-get update && apt-get install -y git && \
    pip install --upgrade pip && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

CMD ["bash", "run_demo.sh"]
