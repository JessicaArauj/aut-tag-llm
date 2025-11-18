#!/bin/bash

# Create backup directory if it doesn't exist
mkdir -p backup

# Resource URL
URL="https://arquivosdadosabertos.saude.gov.br/dados/sisagua/vigilancia_demais_parametros_csv.zip"

# Local file name
FILE="backup/sisagua_resource.zip"

echo "Starting download..."

# Download the ZIP file
curl -L "$URL" -o "$FILE"

echo "Download completed. Extracting files..."

# Extract ZIP into backup folder
unzip -o "$FILE" -d backup/

echo "Extraction finished. All files are stored in the 'backup' directory."