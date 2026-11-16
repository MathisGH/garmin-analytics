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

# --- Frontend bucket: serves the static website (index, anomaly, sleep pages) ---
# A website bucket is described by several Terraform resources: the bucket,
# its website hosting setting, its public-access settings and its policy.

import {
  to = aws_s3_bucket.frontend
  id = "garmin-anomaly-frontend-mathisdurand-2026"
}

resource "aws_s3_bucket" "frontend" {
  bucket = "garmin-anomaly-frontend-mathisdurand-2026"
}

import {
  to = aws_s3_bucket_website_configuration.frontend
  id = "garmin-anomaly-frontend-mathisdurand-2026"
}

resource "aws_s3_bucket_website_configuration" "frontend" {
  bucket = aws_s3_bucket.frontend.id

  index_document {
    suffix = "index.html"
  }
}

import {
  to = aws_s3_bucket_public_access_block.frontend
  id = "garmin-anomaly-frontend-mathisdurand-2026"
}

# All four protections are off on purpose: the site is public. If any of them
# were on, S3 would reject the public-read policy below.
resource "aws_s3_bucket_public_access_block" "frontend" {
  bucket = aws_s3_bucket.frontend.id

  block_public_acls       = false
  ignore_public_acls      = false
  block_public_policy     = false
  restrict_public_buckets = false
}

import {
  to = aws_s3_bucket_policy.frontend
  id = "garmin-anomaly-frontend-mathisdurand-2026"
}

# Anyone may read objects (GetObject only: no listing, no write).
resource "aws_s3_bucket_policy" "frontend" {
  bucket = aws_s3_bucket.frontend.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "PublicReadGetObject"
      Effect    = "Allow"
      Principal = "*"
      Action    = "s3:GetObject"
      Resource  = "${aws_s3_bucket.frontend.arn}/*"
    }]
  })

  # Public access must be allowed before a public policy can be attached
  depends_on = [aws_s3_bucket_public_access_block.frontend]
}