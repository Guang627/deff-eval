from __future__ import annotations

import argparse
import time

import boto3


def build_request(a: argparse.Namespace) -> dict:
    job = f"passk-gen-{a.run_name.replace('.', '-')}-{time.strftime('%Y%m%d-%H%M%S')}"[:63]
    s3_run = f"s3://{a.bucket}/runs/{a.run_name}"
    req: dict = {
        "TrainingJobName": job,
        "RoleArn": a.role,
        "AlgorithmSpecification": {
            "TrainingImage": a.image,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["passk"],
            "ContainerArguments": [
                "generate", "--config", a.config,
                "output_dir=/opt/ml/checkpoints", f"s3_uri={s3_run}", "track=false",
            ],
        },
        "ResourceConfig": {"InstanceType": a.instance_type, "InstanceCount": 1, "VolumeSizeInGB": 100},
        "OutputDataConfig": {"S3OutputPath": f"s3://{a.bucket}/sagemaker-output/"},
        "CheckpointConfig": {"S3Uri": f"{s3_run}/checkpoints", "LocalPath": "/opt/ml/checkpoints"},
        "StoppingCondition": {"MaxRuntimeInSeconds": a.max_hours * 3600},
        "Environment": {"PASSK_DISABLE_TRACKING": "1", "HF_HOME": "/tmp/hf"},
        "Tags": [{"Key": "Project", "Value": "passk-smooth"}],
    }
    if a.spot:
        req["EnableManagedSpotTraining"] = True
        req["StoppingCondition"]["MaxWaitTimeInSeconds"] = a.max_hours * 3600 * 2
    return req


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--role", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--run-name", default="qwen2.5-math-1.5b")
    ap.add_argument("--config", default="configs/generate_qwen_math500.yaml")
    ap.add_argument("--instance-type", default="ml.g5.xlarge")
    ap.add_argument("--max-hours", type=int, default=12)
    ap.add_argument("--no-spot", dest="spot", action="store_false")
    ap.add_argument("--region", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the request and exit")
    a = ap.parse_args()
    req = build_request(a)
    if a.dry_run:
        import json

        print(json.dumps(req, indent=2))
        return
    sm = boto3.client("sagemaker", region_name=a.region)
    sm.create_training_job(**req)
    print(f"started {req['TrainingJobName']}; outputs -> s3://{a.bucket}/runs/{a.run_name}")


if __name__ == "__main__":
    main()
