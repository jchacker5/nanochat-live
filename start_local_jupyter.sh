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
jupyter notebook \
    --NotebookApp.allow_origin='https://colab.research.google.com' \
    --port=8888 \
    --NotebookApp.port_retries=0 \
    --no-browser \
    --notebook-dir="$(pwd)"

