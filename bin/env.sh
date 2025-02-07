#!/bin/bash

# Get the absolute path of the project root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add bin directory to PATH if not already included
if [[ ":$PATH:" != *":$ROOT_DIR/bin:"* ]]; then
    export PATH="$ROOT_DIR/bin:$PATH"
    echo "Added $ROOT_DIR/bin to PATH"
else
    echo "$ROOT_DIR/bin is already in PATH"
fi

