#!/bin/bash
gunicorn --bind 0.0.0.0:6666 wsgi:app