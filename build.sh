#!/bin/bash

# Create build directory
mkdir -p build

# Copy template files
cp -r templates/* build/

# Create index.html with environment variables
sed -i '' "s/'1c13d18f-bdb2-4195-9b54-5bd3d7f2be21'/'${VAPI_PUBLIC_KEY:-1c13d18f-bdb2-4195-9b54-5bd3d7f2be21}'/" build/index.html
sed -i '' "s/'a0f2248e-143f-43a0-8130-7f386a16a94a'/'${ASSISTANT_ID:-a0f2248e-143f-43a0-8130-7f386a16a94a}'/" build/index.html

# Make script executable
chmod +x build.sh
