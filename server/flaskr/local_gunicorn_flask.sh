#!/bin/bash
# gunicorn --bind 0.0.0.0:8888 wsgi:app

gunicorn wsgi:app --timeout=0 --preload --workers=1 --threads=4 --bind=0.0.0.0:8080