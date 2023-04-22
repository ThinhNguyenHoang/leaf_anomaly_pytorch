FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest
ENV PYTHONDONTWRITEBYTECODE=1

# Copies the trainer code to the docker image.
COPY . $HOME/src
WORKDIR $HOME/src

ENTRYPOINT ["python", "main.py", "--gpu", "0", "-bs", "4", "-enc", "wide_resnet50_2", "--pro", "-inp", "256", "--dataset", "plant_village", "--class-name", "Pepper", "--meta-epochs", "1", "--sub-epochs", "1","--gcp","--local-test"]
