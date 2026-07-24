#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh && conda activate proxy_server
set -x
# Create a dedicated directory
mkdir /scratch/wiki_local
cd /scratch/wiki_local

# Download the compressed XML dump (approx. 22GB)
# This will take a while depending on your internet connection.
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# Install WikiExtractor
pip install wikiextractor

# Run the extraction process
# This will take several hours and will output folders (AA, AB, AC, etc.) containing JSON files.
python -m wikiextractor.WikiExtractor enwiki-latest-pages-articles.xml.bz2 --json --output /scratch/wiki_local/extracted_json/t