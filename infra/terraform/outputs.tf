output "bucket" {
  value = aws_s3_bucket.results.bucket
}

output "ecr_repository_url" {
  value = aws_ecr_repository.passk.repository_url
}

output "code_s3_uri" {
  value = local.s3_code_uri
}

output "run_s3_uri" {
  value = local.s3_run_uri
}

output "sagemaker_role_arn" {
  value = aws_iam_role.sagemaker.arn
}

output "instance_id" {
  value = try(aws_instance.generate[0].id, null)
}
