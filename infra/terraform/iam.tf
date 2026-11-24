data "aws_iam_policy_document" "job" {
  statement {
    sid       = "ResultsBucket"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.results.arn, "${aws_s3_bucket.results.arn}/*"]
  }
  statement {
    sid       = "SelfTerminate"
    actions   = ["ec2:TerminateInstances"]
    resources = ["arn:aws:ec2:${var.region}:${data.aws_caller_identity.me.account_id}:instance/*"]
    condition {
      test     = "StringEquals"
      variable = "aws:ResourceTag/Project"
      values   = [var.project]
    }
  }
  statement {
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    sid       = "EcrPull"
    actions   = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"]
    resources = [aws_ecr_repository.passk.arn]
  }
  dynamic "statement" {
    for_each = var.hf_token_ssm_parameter == "" ? [] : [1]
    content {
      sid       = "HfToken"
      actions   = ["ssm:GetParameter"]
      resources = ["arn:aws:ssm:${var.region}:${data.aws_caller_identity.me.account_id}:parameter/${trimprefix(var.hf_token_ssm_parameter, "/")}"]
    }
  }
}

resource "aws_iam_policy" "job" {
  name   = "${var.project}-job"
  policy = data.aws_iam_policy_document.job.json
}

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ec2" {
  name               = "${var.project}-ec2"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

resource "aws_iam_role_policy_attachment" "ec2_job" {
  role       = aws_iam_role.ec2.name
  policy_arn = aws_iam_policy.job.arn
}

resource "aws_iam_role_policy_attachment" "ec2_ssm" {
  role       = aws_iam_role.ec2.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "ec2" {
  name = "${var.project}-ec2"
  role = aws_iam_role.ec2.name
}

data "aws_iam_policy_document" "sm_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker" {
  name               = "${var.project}-sagemaker"
  assume_role_policy = data.aws_iam_policy_document.sm_assume.json
}

resource "aws_iam_role_policy_attachment" "sm_job" {
  role       = aws_iam_role.sagemaker.name
  policy_arn = aws_iam_policy.job.arn
}

data "aws_iam_policy_document" "sm_logs" {
  statement {
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams", "cloudwatch:PutMetricData"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "sm_logs" {
  role   = aws_iam_role.sagemaker.id
  policy = data.aws_iam_policy_document.sm_logs.json
}
