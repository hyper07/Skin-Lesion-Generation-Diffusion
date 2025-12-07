"""
Utility functions for checking and downloading datasets
"""

import os
import sys
import zipfile
from pathlib import Path
import re
from urllib.parse import urlparse, parse_qs

# Try to import requests
try:
    import requests
except ImportError:
    requests = None
    print("Warning: requests library not found. Install it with: pip install requests")


def extract_file_id(url):
    """Extract Google Drive file ID from various URL formats."""
    patterns = [
        r'/file/d/([a-zA-Z0-9-_]+)',
        r'id=([a-zA-Z0-9-_]+)',
        r'/open\?id=([a-zA-Z0-9-_]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None


def download_from_google_drive(file_id, destination):
    """Download a file from Google Drive using its file ID."""
    if requests is None:
        print("Error: requests library is required for downloading.")
        print("Install it with: pip install requests")
        return False
    
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    try:
        print(f"Downloading dataset from Google Drive (ID: {file_id})...")
        
        session = requests.Session()
        response = session.get(url, stream=True)
        
        # Check if we got an HTML response (virus scan warning for large files)
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type:
            print("Large file detected - handling virus scan confirmation...")
            
            html_content = response.text
            
            uuid_match = re.search(r'name="uuid" value="([^"]+)"', html_content)
            confirm_match = re.search(r'name="confirm" value="([^"]+)"', html_content)
            
            if uuid_match and confirm_match:
                uuid_value = uuid_match.group(1)
                confirm_value = confirm_match.group(1)
                
                download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm={confirm_value}&uuid={uuid_value}"
                
                print("Getting download with virus scan bypass...")
                response = session.get(download_url, stream=True)
            else:
                if 'confirm=' in response.url:
                    for key, value in parse_qs(urlparse(response.url).query).items():
                        if key == 'confirm':
                            token = value[0]
                            break
                    url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={token}"
                    response = session.get(url, stream=True)
        
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type:
            print("Error: Still receiving HTML response. The file might be too large or have restricted access.")
            return False
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end='', flush=True)
        
        print(f"\nDownload completed: {destination}")
        return True
        
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """Extract a ZIP file to the specified directory."""
    try:
        print(f"Extracting {zip_path} to {extract_to}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            os.makedirs(extract_to, exist_ok=True)
            zip_ref.extractall(extract_to)
            
            extracted_files = zip_ref.namelist()
            print(f"Extracted {len(extracted_files)} files")
        
        print(f"Extraction completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error extracting ZIP file: {e}")
        return False


def check_dataset_files(dataset_dir):
    """
    Check if required dataset files and folders exist.
    Handles both single folder and split folder structures.
    Also checks in common subdirectories like 'dataverse_files'.
    
    Returns:
        tuple: (all_exist, missing_items)
    """
    dataset_path = Path(dataset_dir)
    missing = []
    
    # Check metadata file - could be in root or subdirectory
    metadata_paths = [
        dataset_path / 'HAM10000_metadata.csv',
        dataset_path / 'dataverse_files' / 'HAM10000_metadata',
        dataset_path / 'dataverse_files' / 'HAM10000_metadata.csv',
    ]
    metadata_found = any(p.exists() for p in metadata_paths)
    if not metadata_found:
        missing.append(('metadata', 'HAM10000_metadata.csv'))
    
    # Check HAM images - could be single folder or split into part_1 and part_2
    # Also check in dataverse_files subdirectory
    ham_images = dataset_path / 'HAM10000_images'
    ham_part1 = dataset_path / 'HAM10000_images_part_1'
    ham_part2 = dataset_path / 'HAM10000_images_part_2'
    ham_part1_sub = dataset_path / 'dataverse_files' / 'HAM10000_images_part_1'
    ham_part2_sub = dataset_path / 'dataverse_files' / 'HAM10000_images_part_2'
    
    ham_found = (ham_images.exists() or 
                 (ham_part1.exists() and ham_part2.exists()) or
                 (ham_part1_sub.exists() and ham_part2_sub.exists()))
    if not ham_found:
        missing.append(('ham_images', 'HAM10000_images or HAM10000_images_part_1/part_2'))
    
    # Check BCN metadata
    bcn_metadata_paths = [
        dataset_path / 'ISIC_metadata.csv',
        dataset_path / 'dataverse_files' / 'ISIC_metadata.csv',
    ]
    bcn_metadata_found = any(p.exists() for p in bcn_metadata_paths)
    if not bcn_metadata_found:
        missing.append(('bcn_metadata', 'ISIC_metadata.csv'))
    
    # Check BCN images
    bcn_images_paths = [
        dataset_path / 'ISIC_images',
        dataset_path / 'dataverse_files' / 'ISIC_images',
    ]
    bcn_images_found = any(p.exists() for p in bcn_images_paths)
    if not bcn_images_found:
        missing.append(('bcn_images', 'ISIC_images'))
    
    return len(missing) == 0, missing


def ensure_dataset(dataset_dir='dataset', google_drive_file_id='1KCjAyv1vjB1p92YcMS1rfTcXzFFc4LT0'):
    """
    Ensure dataset files exist, download and extract if missing.
    
    Args:
        dataset_dir: Directory where dataset should be
        google_drive_file_id: Google Drive file ID for the dataset zip
    
    Returns:
        bool: True if dataset is ready, False otherwise
    """
    dataset_path = Path(dataset_dir)
    dataset_path.mkdir(exist_ok=True)
    
    # Check if files exist
    all_exist, missing = check_dataset_files(dataset_dir)
    
    if all_exist:
        print("✓ All required dataset files found.")
        return True
    
    print(f"Missing dataset files: {[m[0] for m in missing]}")
    print("Downloading dataset from Google Drive...")
    
    # Download the dataset
    zip_filename = f"dataset_{google_drive_file_id}.zip"
    zip_path = dataset_path / zip_filename
    
    # Check if zip already exists
    if not zip_path.exists():
        if not download_from_google_drive(google_drive_file_id, zip_path):
            print("Failed to download dataset.")
            return False
    else:
        print(f"Found existing zip file: {zip_path}")
    
    # Extract if it's a zip file
    if zipfile.is_zipfile(zip_path):
        if not extract_zip(zip_path, dataset_path):
            print("Failed to extract dataset.")
            return False
    else:
        print(f"Downloaded file is not a ZIP archive: {zip_path}")
        return False
    
    # Check again after extraction
    all_exist, missing = check_dataset_files(dataset_dir)
    
    if all_exist:
        print("✓ Dataset ready!")
        # Optionally remove zip file to save space
        # zip_path.unlink()
        return True
    else:
        print(f"Warning: Some files still missing after extraction: {[m[0] for m in missing]}")
        print("Please check the extracted files manually.")
        return False

