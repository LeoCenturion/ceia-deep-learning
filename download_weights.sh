#!/bin/bash

# Script to download and unzip weights and history from Google Drive

# Exit immediately if a command exits with a non-zero status.
set -e

# Google Drive File ID from the shared link
<<<<<<< HEAD
FILE_ID="1ytN-kDuSOc9vsCisEAer9RG7tdhVvZqc"
# Output filename for the downloaded zip
OUTPUT_ZIP="weights-and-history.zip"
# Google Drive download URL format
DOWNLOAD_URL="https://drive.google.com/file/d/1ytN-kDuSOc9vsCisEAer9RG7tdhVvZqc/view?usp=drive_link"
=======
FILE_ID="1N8jtxU7mJbeIIYt8MvAKU5wDDYFWHxC3"
# Output filename for the downloaded zip
OUTPUT_ZIP="weights-and-history.zip"
# Google Drive download URL format
DOWNLOAD_URL="https://drive.google.com/uc?export=download&id=${FILE_ID}"
>>>>>>> 9d71e3b (Refactor)

echo "Downloading ${OUTPUT_ZIP} from Google Drive..."
# Use wget to download. -O specifies the output file.
# The --no-check-certificate flag might be needed if there are SSL issues,
# but try without it first. Add it back if download fails.
# Use --load-cookies /tmp/cookies.txt --save-cookies /tmp/cookies.txt --keep-session-cookies if needed for confirmation pages
wget --load-cookies /tmp/cookies.txt --save-cookies /tmp/cookies.txt --keep-session-cookies -O "${OUTPUT_ZIP}" "${DOWNLOAD_URL}"

# Check if download was successful
if [ $? -ne 0 ]; then
    echo "Error: Download failed. Please check the URL and your internet connection."
    # Attempt to clean up potentially incomplete file
    rm -f "${OUTPUT_ZIP}"
    # Clean up cookies file
    rm -f /tmp/cookies.txt
    exit 1
fi

# Clean up cookies file
rm -f /tmp/cookies.txt

echo "Download complete: ${OUTPUT_ZIP}"

echo "Unzipping ${OUTPUT_ZIP}..."
# Use unzip. -o overwrites files without prompting.
unzip -o "${OUTPUT_ZIP}"

# Check if unzip was successful
if [ $? -ne 0 ]; then
    echo "Error: Unzipping failed. The downloaded file might be corrupted or not a zip file."
    exit 1
fi

echo "Unzipping complete."
echo "Files extracted to the current directory."

# Optional: Remove the zip file after successful extraction
# echo "Removing ${OUTPUT_ZIP}..."
# rm "${OUTPUT_ZIP}"

echo "Script finished successfully."
exit 0
