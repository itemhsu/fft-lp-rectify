#!/bin/bash

# -------------------------------
# Configuration
# -------------------------------

# Path to the directory containing original images
INPUT_DIR="/home/itemhsu/amtk/fft-lp-rectify/img/20241228/218/C++/org"

# Path to the directory where output images will be saved
OUTPUT_DIR="/home/itemhsu/amtk/fft-lp-rectify/img/20241228/218/C++/output"

# Path to the C++ executable
EXECUTABLE="./lp_rect_clean"

# Supported image extensions (add or remove as needed)
EXTENSIONS=("jpg" "jpeg" "png" "bmp" "tiff")

# -------------------------------
# Create Output Directory if it doesn't exist
# -------------------------------
mkdir -p "$OUTPUT_DIR"

# -------------------------------
# Check if Executable Exists
# -------------------------------
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable not found at $EXECUTABLE"
    exit 1
fi

# -------------------------------
# Loop Through Each Image File
# -------------------------------
for EXT in "${EXTENSIONS[@]}"; do
    # Using globbing to match files with the current extension (case-insensitive)
    shopt -s nullglob
    for IMG in "$INPUT_DIR"/*."$EXT" "$INPUT_DIR"/*."${EXT^^}"; do
        # Check if the glob didn't match any files
        if [ ! -e "$IMG" ]; then
            continue
        fi

        # Extract the filename without the directory
        FILENAME=$(basename "$IMG")

        # Extract the prefix (filename without extension)
        PREFIX="${FILENAME%.*}"

        echo "Processing $FILENAME..."
        echo "$INPUT_DIR/$FILENAME"
        myIMG="${INPUT_DIR}/${FILENAME}"
        echo "$myIMG"

        # Construct the full output path
        OUTPUT_FILE="$OUTPUT_DIR/$PREFIX.jpg"

        # Execute the C++ program with full paths
        "$EXECUTABLE" "$myIMG"

        # Check if the output file was created
        if [ -f "$OUTPUT_FILE" ]; then
            echo "Saved processed image as $OUTPUT_FILE"
        else
            echo "Warning: Output file $OUTPUT_FILE was not created."
        fi
    done
    shopt -u nullglob
done

echo "All images have been processed."
