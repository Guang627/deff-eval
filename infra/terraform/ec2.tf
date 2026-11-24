data "aws_ssm_parameter" "dlami" {
  name = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
}

data "aws_vpc" "default" {
  default = true
}

resource "aws_security_group" "worker" {
  name        = "${var.project}-worker"
  description = "egress only; access via SSM Session Manager"
  vpc_id      = data.aws_vpc.default.id
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "generate" {
  count                  = var.launch_generation ? 1 : 0
  ami                    = data.aws_ssm_parameter.dlami.value
  instance_type          = var.instance_type
  iam_instance_profile   = aws_iam_instance_profile.ec2.name
  vpc_security_group_ids = [aws_security_group.worker.id]
  instance_initiated_shutdown_behavior = var.use_spot ? null : "terminate"

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        spot_instance_type             = "one-time"
        instance_interruption_behavior = "terminate"
      }
    }
  }

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
    encrypted   = true
  }

  metadata_options {
    http_tokens = "required"
    http_put_response_hop_limit = 2
    http_endpoint               = "enabled"
  }

  user_data_replace_on_change = true
  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    region       = var.region
    code_source  = var.code_source
    vllm_image   = var.vllm_image
    image        = "${aws_ecr_repository.passk.repository_url}:${var.image_tag}"
    registry     = split("/", aws_ecr_repository.passk.repository_url)[0]
    config       = var.generate_config
    overrides    = var.generate_overrides
    s3_uri       = local.s3_run_uri
    s3_code_uri  = local.s3_code_uri
    hf_token_ssm = var.hf_token_ssm_parameter
  })

  tags = { Name = "${var.project}-generate-${var.run_name}" }
}
