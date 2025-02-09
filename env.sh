#!/bin/bash

# Get the absolute path of the project root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add bin directory to PATH if not already included
export PATH="$ROOT_DIR/bin:$PATH"
