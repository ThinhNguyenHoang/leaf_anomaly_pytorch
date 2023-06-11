#!/bin/bash

# Prepare log files and start outputting logs to stdout
mkdir -p ./code/logs
touch ./code/logs/gunicorn.log
touch ./code/logs/gunicorn-access.log
tail -n 0 -f ./code/logs/gunicorn*.log &

# Setting environment variable
export DJANGO_SETTINGS_MODULE=django_docker_azure.settings

exec gunicorn wsgi:app --timeout=0 --preload --workers=1 --threads=4 --bind=0.0.0.0:8080 \
    --log-level=info \
    --log-file=./code/logs/gunicorn.log \
    --access-logfile=./code/logs/gunicorn-access.log \
"$@"