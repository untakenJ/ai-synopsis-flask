#!/bin/bash

# Quick script to build and run the web server
# Usage: ./docker-web-server.sh [port]

set -e

PORT=${1:-5000}
CONTAINER_NAME="ai-synopsis-web"

echo "Building and starting AI Synopsis web server on port $PORT..."

# Build and run web server
./docker-build-and-run.sh --mode web-server --port $PORT --name $CONTAINER_NAME

echo ""
echo "Web server is running!"
echo "Access the application at: http://localhost:$PORT"
echo "API status: http://localhost:$PORT/api-status"
echo ""
echo "To view logs: docker logs $CONTAINER_NAME"
echo "To stop: docker stop $CONTAINER_NAME"
echo "To remove: docker rm $CONTAINER_NAME"
