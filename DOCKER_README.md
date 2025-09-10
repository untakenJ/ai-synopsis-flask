# Docker Setup for AI Synopsis Flask

This directory contains Docker configuration files for the AI Synopsis Flask application with support for multiple deployment modes.

## Files

- `Dockerfile` - Multi-stage Docker configuration supporting two build targets
- `run_single_refresh.py` - Python script that executes a single cache refresh
- `docker-build-and-run.sh` - Main script to build and run containers with different modes
- `docker-web-server.sh` - Quick script to build and run the web server
- `docker-refresh-cache.sh` - Quick script to build and run a single cache refresh
- `.dockerignore` - Files to exclude from Docker build context

## Build Targets

The Dockerfile supports two build targets:

1. **`web-server`** - Full web application with Flask and Gunicorn
2. **`refresh-cache`** - Single cache refresh operation that exits after completion

## Prerequisites

1. Docker installed on your system
2. Environment variables configured (see below)

## Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# API Keys
DEEPSEEK_API_KEY=your_deepseek_api_key
DIFY_API_KEY=your_dify_api_key
OPENAI_API_KEY=your_openai_api_key
API_KEY=your_api_key

# AWS S3 Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=your_s3_bucket_name
S3_BUCKET_URL=your_s3_bucket_url  # Optional

# Application Configuration
CACHE_DURATION_HOURS=48
TARGET_URLS=https://tmz.com,https://bbc.com,https://cnn.com,https://espn.com,https://pagesix.com/
MAX_TITLES_PER_URL=40
MAX_TITLES_OVERALL=80
NUM_HEADLINES=5
```

## Usage

### Quick Start Scripts

#### Web Server
```bash
# Start web server on default port 5000
./docker-web-server.sh

# Start web server on custom port
./docker-web-server.sh 8080
```

#### Cache Refresh
```bash
# Run single cache refresh
./docker-refresh-cache.sh
```

### Advanced Usage

#### Using the main script with options

```bash
# Show help
./docker-build-and-run.sh --help

# Build and run web server
./docker-build-and-run.sh --mode web-server --port 8080

# Build and run cache refresh
./docker-build-and-run.sh --mode refresh-cache

# Custom container name
./docker-build-and-run.sh --mode web-server --name my-app --port 3000
```

#### Manual Docker commands

**Web Server:**
```bash
# Build web server image
docker build --target web-server -t ai-synopsis-web-server .

# Run web server
docker run -d --name ai-synopsis-web -p 5000:5000 --env-file .env ai-synopsis-web-server
```

**Cache Refresh:**
```bash
# Build refresh cache image
docker build --target refresh-cache -t ai-synopsis-refresh-cache .

# Run cache refresh
docker run --name ai-synopsis-refresh --env-file .env ai-synopsis-refresh-cache
```

#### Run with specific environment variables

```bash
docker run --name ai-synopsis-web \
  -p 5000:5000 \
  -e DEEPSEEK_API_KEY=your_key \
  -e DIFY_API_KEY=your_key \
  -e OPENAI_API_KEY=your_key \
  -e API_KEY=your_key \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_key \
  -e AWS_REGION=us-east-1 \
  -e S3_BUCKET_NAME=your_bucket \
  ai-synopsis-web-server
```

## What each container does

### Web Server Container (`web-server` target)
1. **Starts Flask application**: Runs the full web API with all endpoints
2. **Uses Gunicorn**: Production-ready WSGI server with multiple workers
3. **Health checks**: Built-in health monitoring
4. **Persistent service**: Runs continuously until stopped
5. **Port exposure**: Accessible on configured port (default 5000)

### Cache Refresh Container (`refresh-cache` target)
1. **Analyzes news sources**: Fetches and analyzes news from configured URLs
2. **Verifies news items**: Uses AI to verify and summarize news items
3. **Generates images**: Creates images for today's news items using DALL-E
4. **Saves to cache**: Stores results in S3 cache for the web application
5. **Exits**: Container terminates after completing the operation

## Output

The container will output:
- Progress messages during analysis
- Summary of found news items
- Top 5 news items with titles and summaries
- Success/failure status

## Troubleshooting

### Common issues:

1. **Missing environment variables**: Ensure all required environment variables are set
2. **API key issues**: Verify that all API keys are valid and have proper permissions
3. **S3 access issues**: Check AWS credentials and S3 bucket permissions
4. **Network issues**: Ensure the container can access external APIs

### Debug mode:

To run with more verbose output, you can modify the `run_single_refresh.py` script or add debug environment variables.

## Deployment Scenarios

### Web Server Deployment
- **Production**: Use the `web-server` target for full application deployment
- **Load balancing**: Can run multiple instances behind a load balancer
- **Container orchestration**: Suitable for Kubernetes, Docker Swarm, etc.
- **Health monitoring**: Built-in health checks for container orchestration

### Cache Refresh Scheduling
The `refresh-cache` target is designed to be run as a scheduled job:
- **Cron jobs**: Run periodically to update cache
- **Kubernetes CronJob**: Scheduled container execution
- **CI/CD pipelines**: Cache updates during deployment
- **AWS Lambda**: Serverless cache refresh (with appropriate runtime)
- **Exit codes**: 0 for success, 1 for failure

## Resource requirements

- **Memory**: At least 1GB RAM recommended
- **CPU**: 1-2 cores recommended
- **Storage**: Minimal (only for temporary files during execution)
- **Network**: Access to external APIs (DeepSeek, OpenAI, Dify, news websites)
