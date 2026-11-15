terraform {
  required_version = ">= 1.5" # import blocks need Terraform 1.5 or newer

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0" # any 6.x release, never 7.0
    }
  }
}

provider "aws" {
  region = "eu-west-3"
}

# --- Data bucket: holds the precomputed JSON files read by the Lambda function ---
# The bucket in the region 'eu-north-1" is a mistake I did when I created it,
# I keep it like that for now, so that's why I need to specify it like that

import {
  to = aws_s3_bucket.data
  id = "garmin-anomaly-mathisdurand-2026@eu-north-1"
}

resource "aws_s3_bucket" "data" {
  region = "eu-north-1"
  bucket = "garmin-anomaly-mathisdurand-2026"
}