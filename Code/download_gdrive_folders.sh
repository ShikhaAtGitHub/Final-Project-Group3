#!/bin/bash

# Function to get the download link
get_gdrive_download_link() {
    CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
    wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
    rm -rf /tmp/cookies.txt
}

# Download the first folder
get_gdrive_download_link 18cXGLbLFhK4IUDCutKjFiq18D0kOIw69 model.zip

# Download the second folder
get_gdrive_download_link 17g_i4aZemoDlHzdVGsS-SrlYUr8AcnkX tokenizer.zip
