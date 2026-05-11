FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    scikit-learn==1.6.0 \
    pandas \
    numpy \
    cloudpickle

COPY model/ /app/model/

EXPOSE 5001

ENTRYPOINT ["mlflow", "models", "serve", "--no-conda", "-m", "/app/model", "-h", "0.0.0.0", "-p", "5001"]
