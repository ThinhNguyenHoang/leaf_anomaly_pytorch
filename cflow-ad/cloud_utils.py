import os

BUCKET_NAME = os.environ.get('BUCKET_NAME') or 'thesis-data-bucket'
MODEL_RUNNAME = 'leaf_anomaly_det'
def get_bucket_prefix(bucket_name=BUCKET_NAME):
    return f'/gcs/{bucket_name}/{MODEL_RUNNAME}'

BUCKET_PREFIX = f'/gcs/{BUCKET_NAME}'