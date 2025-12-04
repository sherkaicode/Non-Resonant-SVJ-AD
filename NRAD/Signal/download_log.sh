#!/bin/bash
# ---------------------------------------------------------------
# CERN LXPLUS file downloader (interactive)
# Author: fjumawan
# ---------------------------------------------------------------

USER="fjumawan"
BASE="/eos/user/f/fjumawan/SWAN_projects/Research_HEP/Signal"
DEST="./cross_section_info"

# Create local download directory if missing
mkdir -p "$DEST"

# Ask user for the setting tag (e.g., 1000_2)
read -p "Enter the setting (e.g., 1000_2): " tag

# Construct full EOS path
REMOTE_PATH="${BASE}/${tag}/log.generate"
LOCAL_FILE="${DEST}/${tag}.log"

echo "-------------------------------------------------"
echo "Downloading:"
echo "  From: ${REMOTE_PATH}"
echo "  To:   ${LOCAL_FILE}"
echo "-------------------------------------------------"

# Run scp to copy the file
scp "${USER}@lxplus.cern.ch:${REMOTE_PATH}" "${LOCAL_FILE}"

# Check if scp succeeded
if [ $? -eq 0 ]; then
    echo "✅ Download complete: ${LOCAL_FILE}"
else
    echo "❌ Download failed. Please check the tag or your connection."
fi
