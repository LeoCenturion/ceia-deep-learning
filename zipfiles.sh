#!/bin/bash

# Define the name of the zip file
zip_filename="weights-and-history.zip"

# Find all files ending with .pth or .csv and zip them
find . -maxdepth 1 -type f \( -name "*.pth" -o -name "*.csv" \)  | zip -0 "$zip_filename" -@

echo "Successfully zipped all .pth and .csv files in the current directory (and subdirectories) to '$zip_filename'"
