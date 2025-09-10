#!/bin/bash

# Docker build and run script for AI Synopsis Flask
# Supports two modes: web-server and refresh-cache

set -e

# Default configuration
MODE="refresh-cache"
IMAGE_NAME="ai-synopsis"
CONTAINER_NAME="ai-synopsis-container"
PORT="5000"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --mode MODE        Build mode: 'web-server' or 'refresh-cache' (default: refresh-cache)"
    echo "  -p, --port PORT        Port for web server (default: 5000)"
    echo "  -n, --name NAME        Container name (default: ai-synopsis-container)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --mode web-server --port 8080"
    echo "  $0 --mode refresh-cache"
    echo "  $0 -m web-server -p 5000"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -n|--name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate mode
if [[ "$MODE" != "web-server" && "$MODE" != "refresh-cache" ]]; then
    echo "Error: Invalid mode '$MODE'. Must be 'web-server' or 'refresh-cache'"
    exit 1
fi

# Set image name based on mode
IMAGE_NAME="ai-synopsis-$MODE"

echo "Building Docker image: $IMAGE_NAME (mode: $MODE)"

# Build the Docker image with target
docker build --target $MODE -t $IMAGE_NAME .

echo "Docker image built successfully."

# Check if container already exists and remove it
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Removing existing container: $CONTAINER_NAME"
    docker rm $CONTAINER_NAME
fi

# Run the container based on mode
if [ -f .env ]; then
    echo "Using environment variables from .env file"
    
    if [[ "$MODE" == "web-server" ]]; then
        echo "Starting web server on port $PORT..."
        docker run -d --name $CONTAINER_NAME -p $PORT:5000 --env-file .env $IMAGE_NAME
        
        echo "Web server started successfully!"
        echo "Access the application at: http://localhost:$PORT"
        echo "API status: http://localhost:$PORT/api-status"
        echo ""
        echo "To view logs: docker logs $CONTAINER_NAME"
        echo "To stop: docker stop $CONTAINER_NAME"
        echo "To remove: docker rm $CONTAINER_NAME"
        
    else
        echo "Running single cache refresh operation..."
        docker run --name $CONTAINER_NAME --env-file .env $IMAGE_NAME
        
        # Check exit code
        if [ $? -eq 0 ]; then
            echo "Single refresh operation completed successfully!"
        else
            echo "Single refresh operation failed!"
            exit 1
        fi
        
        # Clean up container
        echo "Cleaning up container: $CONTAINER_NAME"
        docker rm $CONTAINER_NAME
    fi
    
else
    echo "Warning: .env file not found. Make sure to set environment variables manually."
    echo "Required environment variables:"
    echo "  - DEEPSEEK_API_KEY"
    echo "  - DIFY_API_KEY"
    echo "  - OPENAI_API_KEY"
    echo "  - API_KEY"
    echo "  - AWS_ACCESS_KEY_ID"
    echo "  - AWS_SECRET_ACCESS_KEY"
    echo "  - AWS_REGION"
    echo "  - S3_BUCKET_NAME"
    echo "  - S3_BUCKET_URL (optional)"
    echo ""
    echo "Example:"
    echo "docker run --name $CONTAINER_NAME -e DEEPSEEK_API_KEY=your_key -e ... $IMAGE_NAME"
    exit 1
fi

echo "Done."
