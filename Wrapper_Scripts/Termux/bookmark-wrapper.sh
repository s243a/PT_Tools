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

# Check if output directory exists and prompt to create it if needed
if [ ! -d "$OUTPUT_DIR" ]; then
  echo "Output directory '$OUTPUT_DIR' does not exist."
  read -p "Would you like to create this directory? (y/n): " answer
  
  case ${answer:0:1} in
    y|Y )
      echo "Creating output directory..."
      mkdir -p "$OUTPUT_DIR" || {
        echo "Error: Failed to create output directory. Please check permissions."
        exit 1
      }
      ;;
    * )
      echo "Aborted. Please create the directory manually before running this script."
      exit 1
      ;;
  esac
fi

# Generate a unique timestamp for this run
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
TEMP_SUBDIR="${TEMP_DIR}/${TIMESTAMP}"

# Create the specific temporary directory to use
mkdir -p "$TEMP_SUBDIR"

# Run the Python script with the specified temp directory and output directory
if [ $# -gt 1 ]; then
  # Custom mapping was provided
  python "$PYTHON_SCRIPT" "$URL" --favorites-dir "$TEMP_SUBDIR" --output-dir "$OUTPUT_DIR" --mapping "$2"
else
  # No custom mapping
  python "$PYTHON_SCRIPT" "$URL" --favorites-dir "$TEMP_SUBDIR" --output-dir "$OUTPUT_DIR"
fi

# Check if the Python script was successful
if [ $? -ne 0 ]; then
  echo "Error: The Python script failed. Temporary files will not be deleted."
  exit 1
fi

# Clean up safety check - prevent accidental deletion by verifying the path
TEMP_SUBDIR_REAL=$(realpath "$TEMP_SUBDIR")
EXPECTED_PATH_PREFIX="/data/data/com.termux/files/home/favorites/$TIMESTAMP"

if [[ "$TEMP_SUBDIR_REAL" != "$EXPECTED_PATH_PREFIX"* ]]; then
  echo "Error: Temporary directory path mismatch. Cleanup aborted for safety."
  echo "Expected path to start with: $EXPECTED_PATH_PREFIX"
  echo "Found: $TEMP_SUBDIR_REAL"
  exit 1
fi

# Safer cleanup - first confirm the directory exists and is under our control
if [ -d "$TEMP_SUBDIR" ]; then
  echo "Cleaning up temporary files..."
  
  # Check if find is available
  if command -v find >/dev/null 2>&1; then
    # Primary method: Use find to delete files first (safer than recursive directory removal)
    find "$TEMP_SUBDIR" -type f | sort -r | while read file; do
      echo "Removing file: $file"
      rm "$file"
    done
    
    # Then find and remove empty directories
    find "$TEMP_SUBDIR" -type d | sort -r | while read dir; do
      if [ "$dir" != "$TEMP_SUBDIR" ]; then  # Skip the parent directory for now
        echo "Removing directory: $dir"
        rmdir "$dir" 2>/dev/null || true
      fi
    done
    
    # Finally remove the parent directory
    echo "Removing parent directory: $TEMP_SUBDIR"
    rmdir "$TEMP_SUBDIR" 2>/dev/null || {
      echo "Note: Could not remove $TEMP_SUBDIR cleanly, using rm -rf as fallback"
      rm -rf "$TEMP_SUBDIR"
    }
  else
    # Fallback method: Use rm -rf if find isn't available
    echo "Note: 'find' utility not available, using fallback removal method"
    echo "Removing directory: $TEMP_SUBDIR"
    rm -rf "$TEMP_SUBDIR"
  fi
else
  echo "Warning: Temporary directory $TEMP_SUBDIR does not exist or is not a directory."
fi

# Show the output file location
echo ""
echo "Bookmarks have been saved to: $OUTPUT_DIR"
echo "Temporary files have been cleaned up."