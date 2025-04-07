#!/bin/bash
# bookmark-wrapper.sh
# Wrapper script for combined-bookmarks.py that sets output directory
# and cleans up temporary .url files after completion

# Check if a URL was provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <url> [custom_mapping]"
  echo "Example: $0 https://en.wikipedia.org/wiki/Python_(programming_language)"
  echo "Example with custom mapping: $0 https://en.wikipedia.org/wiki/Python_(programming_language) \"See also:Related,Categories:Topics\""
  exit 1
fi

# Set paths
URL="$1"
PYTHON_SCRIPT="combined-bookmarks.py"
TEMP_DIR="/data/data/com.termux/files/home/favorites"
OUTPUT_DIR="/data/data/com.termux/files/home/storage/documents/bookmarks"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Record starting timestamp to identify created directories
START_TIME=$(date +%Y-%m-%d_%H-%M-%S)

# Run the Python script with the specified output directory
if [ $# -gt 1 ]; then
  # Custom mapping was provided
  python "$PYTHON_SCRIPT" "$URL" --favorites-dir "$TEMP_DIR" --output-dir "$OUTPUT_DIR" --mapping "$2"
else
  # No custom mapping
  python "$PYTHON_SCRIPT" "$URL" --favorites-dir "$TEMP_DIR" --output-dir "$OUTPUT_DIR"
fi

# Check if the Python script was successful
if [ $? -ne 0 ]; then
  echo "Error: The Python script failed. Temporary files will not be deleted."
  exit 1
fi

# Find directories created after the start time
echo "Cleaning up temporary files..."
for dir in $(find "$TEMP_DIR" -type d -mindepth 1 -newermt "$START_TIME"); do
  echo "Removing directory: $dir"
  rm -rf "$dir"
done

# Show the output file location
echo ""
echo "Bookmarks have been saved to: $OUTPUT_DIR"
echo "Temporary files have been cleaned up."