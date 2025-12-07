# Dataset Directory

This directory contains the project's datasets.

## Downloading Data from Google Drive

Use the `download_dataset.py` script in the project root to download and extract datasets from Google Drive:

### Usage:
```bash
python download_dataset.py <google_drive_url_or_file_id> [output_filename]
```

### Examples:

1. Download using a Google Drive sharing URL:
```bash
python download_dataset.py "https://drive.google.com/file/d/1ABC123DEF456GHI789JKL/view?usp=sharing"
```

2. Download using just the file ID:
```bash
python download_dataset.py 1ABC123DEF456GHI789JKL
```

3. Download with a custom filename:
```bash
python download_dataset.py 1ABC123DEF456GHI789JKL my_dataset.zip
```

### Features:
- Automatically extracts ZIP files after download
- Shows download progress
- Handles Google Drive's virus scan confirmation
- Creates the dataset directory if it doesn't exist
- Option to remove ZIP file after extraction

### Supported URL formats:
- https://drive.google.com/file/d/FILE_ID/view
- https://drive.google.com/open?id=FILE_ID
- https://drive.google.com/uc?id=FILE_ID
- Direct file ID (e.g., 1ABC123DEF456GHI789JKL)

### Requirements:
Make sure you have installed the project dependencies:
```bash
uv sync
```

The script will download files to this dataset/ directory and automatically extract ZIP archives.