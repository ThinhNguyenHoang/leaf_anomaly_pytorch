# FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest
FROM local_jupyter_torch_dev:latest
ENV PYTHONDONTWRITEBYTECODE=1

COPY ./requirements.txt ./
RUN pip install --upgrade pip
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
# Copies the trainer code to the docker image.
COPY . $HOME/src
WORKDIR $HOME/src
# GUNICORN PORT
EXPOSE 6000
# FLASK PORT
EXPOSE 5000
EXPOSE 8080
# Start the flask server
# ENTRYPOINT ["python", "main.py", "--gpu", "0", "-bs", "4", "-enc", "wide_resnet50_2", "--pro", "-inp", "256", "--dataset", "plant_village", "--class-name", "Pepper", "--meta-epochs", "1", "--sub-epochs", "1","--gcp"]
# ENTRYPOINT ["sh", "entrypoint.sh"]
CMD ["gunicorn", "wsgi:app", "--timeout=0", "--preload", \
     "--workers=1", "--threads=4", "--bind=0.0.0.0:8080"]