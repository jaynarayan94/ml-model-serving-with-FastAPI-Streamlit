import os
import boto3

# Initialize S3 client and specify bucket details
bucket_name = "mlops-jaynd"


s3 = boto3.client('s3')

# Function to download model from S3
def download_dir(local_path, model_name):
    s3_prefix = "ml-models/"+model_name
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key = key['Key']
                local_file = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))
                s3.download_file(bucket_name, s3_key, local_file)
