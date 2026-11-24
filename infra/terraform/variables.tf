variable "region" {
  type    = string
  default = "us-east-1"
}

variable "project" {
  type    = string
  default = "passk-smooth"
}

variable "instance_type" {
  description = "GPU instance for vLLM generation. g5.xlarge = 1x A10G 24GB, enough for a 1.5B-7B model."
  type        = string
  default     = "g5.xlarge"
}

variable "use_spot" {
  description = "Run on a spot instance (~60-70% cheaper; generation is resumable, so interruptions only cost time)."
  type        = bool
  default     = true
}

variable "launch_generation" {
  description = "Create the EC2 instance that runs the generation job. Set false to create only bucket/ECR/IAM."
  type        = bool
  default     = false
}

variable "code_source" {
  description = "\"s3\": run the public vLLM image and install this repo from a tarball in S3 (no local Docker needed; upload with scripts/upload_code.sh). \"ecr\": run the image pushed by scripts/push_ecr.sh."
  type        = string
  default     = "s3"
  validation {
    condition     = contains(["s3", "ecr"], var.code_source)
    error_message = "code_source must be \"s3\" or \"ecr\"."
  }
}

variable "vllm_image" {
  description = "Public vLLM image used when code_source = \"s3\"."
  type        = string
  default     = "vllm/vllm-openai:v0.10.0"
}

variable "image_tag" {
  description = "ECR image tag used when code_source = \"ecr\"."
  type        = string
  default     = "gpu-latest"
}

variable "generate_overrides" {
  description = "Extra key=value overrides for `passk generate`, e.g. \"limit=20 n_samples=16\" for a pilot run."
  type        = string
  default     = ""
}

variable "bucket_force_destroy" {
  description = "Allow `terraform destroy` to delete the results bucket even when it holds objects. Download results first."
  type        = bool
  default     = false
}

variable "generate_config" {
  description = "Config file inside the image to run."
  type        = string
  default     = "configs/generate_qwen_math500.yaml"
}

variable "run_name" {
  description = "S3 prefix for this run's outputs."
  type        = string
  default     = "qwen2.5-math-1.5b"
}

variable "hf_token_ssm_parameter" {
  description = "Optional SSM SecureString parameter holding a Hugging Face token (for gated models)."
  type        = string
  default     = ""
}

variable "root_volume_gb" {
  type    = number
  default = 150
}
