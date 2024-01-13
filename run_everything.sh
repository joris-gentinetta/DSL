#!/bin/bash

# Check if two arguments are given
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <filename> <config_id>"
    exit 1
fi

# Assign arguments to variables
FILE=$1
CONFIG_ID=$2

# Run the segmentation script
python segment/main.py --file "$FILE" --config_id "$CONFIG_ID"

# Run the tracking script
python track/track.py --file "$FILE" --config_id "$CONFIG_ID"

# Run the postprocessing script
python track/postprocess.py --file "$FILE" --config_id "$CONFIG_ID"

# Display the results
python display.py --file "$FILE" --config_id "$CONFIG_ID"
