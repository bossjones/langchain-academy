#!/usr/bin/env bash

# Find all .ipynb files and convert them to .py files using marimo
find . -name "*.ipynb" -type f -exec bash -c '
    for file do
        dir=$(dirname "$file")
        filename=$(basename "$file" .ipynb)
        output_file="${dir}/${filename}_marimo.py"
        echo "Converting $file to $output_file"
        marimo convert "$file" > "$output_file"
    done
' bash {} +

echo "Conversion complete!"
