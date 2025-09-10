#!/bin/bash

# Quick script to build and run a single cache refresh
# Usage: ./docker-refresh-cache.sh

set -e

echo "Building and running AI Synopsis cache refresh..."

# Build and run refresh cache
./docker-build-and-run.sh --mode refresh-cache

echo "Cache refresh completed!"
