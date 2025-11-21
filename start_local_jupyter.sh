#!/bin/bash
# Start local Jupyter server for Colab connection
# This allows Colab to use your local machine as runtime

echo "Starting Jupyter server for Colab connection..."
echo ""
echo "After starting, you'll see a URL with a token."
echo "Copy that URL and use it in Colab:"
echo "  Connect > Connect to local runtime > Paste URL"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start Jupyter with Colab-compatible settings
# Try jupyter directly, fall back to python3 -m jupyter
if command -v jupyter &> /dev/null; then
    JUPYTER_CMD=jupyter
else
    JUPYTER_CMD="python3 -m jupyter"
fi

$JUPYTER_CMD notebook \
    --NotebookApp.allow_origin='https://colab.research.google.com' \
    --port=8888 \
    --NotebookApp.port_retries=0 \
    --no-browser \
    --notebook-dir="$(pwd)"

