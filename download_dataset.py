#!/usr/bin/env python3
"""
Script to download a file from Google Drive and extract it to the dataset folder.
Supports both direct download links and Google Drive file IDs.
"""

import os
import sys
import zipfile
from pathlib import Path
import re
from urllib.parse import urlparse, parse_qs

# Try to import requests, install if not available
requests = None
try:
    import requests
except ImportError:
    print("requests library not found. Installing...")
    import subprocess
    try:
        # Try to install with pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
        import requests  # noqa: F401
        print("Successfully installed requests library.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to install requests library: {e}")
        print("Please install it manually using one of these commands:")
        print("  uv add requests")
        print("  pip install requests")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error while installing requests: {e}")
        print("Please install requests manually and try again.")
        sys.exit(1)


def extract_file_id(url):
    """
    Extract Google Drive file ID from various URL formats.
    
    Args:
        url (str): Google Drive URL
        
    Returns:
        str: File ID or None if not found
    """
    # Pattern for file ID in various Google Drive URL formats
    patterns = [
        r'/file/d/([a-zA-Z0-9-_]+)',  # /file/d/FILE_ID/view
        r'id=([a-zA-Z0-9-_]+)',       # ?id=FILE_ID
        r'/open\?id=([a-zA-Z0-9-_]+)' # /open?id=FILE_ID
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None


def download_from_google_drive(file_id, destination):
    """
    Download a file from Google Drive using its file ID.
    
    Args:
        file_id (str): Google Drive file ID
        destination (str): Local file path to save the download
        
    Returns:
        bool: True if download successful, False otherwise
    """
    # Google Drive direct download URL
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    try:
        print(f"Downloading file from Google Drive (ID: {file_id})...")
        
        # Start the download
        session = requests.Session()
        response = session.get(url, stream=True)
        
        # Check if we got an HTML response (virus scan warning for large files)
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type:
            print("Large file detected - handling virus scan confirmation...")
            
            # Get the HTML content to extract the actual download URL
            html_content = response.text
            
            # Look for the download form with confirmation parameters
            import re
            
            # Extract the UUID and confirmation token from the HTML
            uuid_match = re.search(r'name="uuid" value="([^"]+)"', html_content)
            confirm_match = re.search(r'name="confirm" value="([^"]+)"', html_content)
            
            if uuid_match and confirm_match:
                uuid_value = uuid_match.group(1)
                confirm_value = confirm_match.group(1)
                
                # Construct the direct download URL with confirmation
                download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm={confirm_value}&uuid={uuid_value}"
                
                print("Getting download with virus scan bypass...")
                response = session.get(download_url, stream=True)
            else:
                # Fallback: try the old method
                print("Trying alternative confirmation method...")
                # Look for any confirmation token in the URL
                if 'confirm=' in response.url:
                    for key, value in parse_qs(urlparse(response.url).query).items():
                        if key == 'confirm':
                            token = value[0]
                            break
                    url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={token}"
                    response = session.get(url, stream=True)
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Check if we still got HTML (download failed)
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type:
            print("Error: Still receiving HTML response. The file might be too large or have restricted access.")
            print("Try making the file publicly accessible or using a different sharing method.")
            return False
        
        # Get file size from headers if available
        total_size = int(response.headers.get('content-length', 0))
        
        # Download the file
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
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """
    Extract a ZIP file to the specified directory.
    
    Args:
        zip_path (str): Path to the ZIP file
        extract_to (str): Directory to extract files to
        
    Returns:
        bool: True if extraction successful, False otherwise
    """
    try:
        print(f"Extracting {zip_path} to {extract_to}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Create extraction directory if it doesn't exist
            os.makedirs(extract_to, exist_ok=True)
            
            # Extract all files
            zip_ref.extractall(extract_to)
            
            # List extracted files
            extracted_files = zip_ref.namelist()
            print(f"Extracted {len(extracted_files)} files:")
            for file in extracted_files[:10]:  # Show first 10 files
                print(f"  - {file}")
            if len(extracted_files) > 10:
                print(f"  ... and {len(extracted_files) - 10} more files")
        
        print(f"Extraction completed successfully!")
        return True
        
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid ZIP file.")
        return False
    except Exception as e:
        print(f"Error extracting ZIP file: {e}")
        return False


def main():
    """Main function to handle command line arguments and orchestrate the download."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python download_dataset.py <google_drive_url_or_file_id> [output_filename]")
        print("")
        print("Examples:")
        print("  python download_dataset.py https://drive.google.com/file/d/1ABC123/view")
        print("  python download_dataset.py 1ABC123DEF456 dataset.zip")
        print("")
        print("The file will be downloaded to the dataset/ folder and extracted if it's a ZIP file.")
        sys.exit(1)
    
    # Get the Google Drive URL or file ID
    url_or_id = sys.argv[1]
    
    # Extract file ID from URL if needed
    if url_or_id.startswith('http'):
        file_id = extract_file_id(url_or_id)
        if not file_id:
            print("Error: Could not extract file ID from the provided URL.")
            print("Make sure the URL is a valid Google Drive sharing link.")
            sys.exit(1)
    else:
        file_id = url_or_id
    
    # Set up paths
    project_root = Path(__file__).parent
    dataset_dir = project_root / "dataset"
    
    # Create dataset directory if it doesn't exist
    dataset_dir.mkdir(exist_ok=True)
    
    # Determine output filename
    if len(sys.argv) >= 3:
        filename = sys.argv[2]
    else:
        filename = f"download_{file_id}.zip"
    
    download_path = dataset_dir / filename
    
    print(f"Dataset directory: {dataset_dir}")
    print(f"Download path: {download_path}")
    print()
    
    # Download the file
    if download_from_google_drive(file_id, download_path):
        print()
        
        # Check if the downloaded file is a ZIP and extract it
        if zipfile.is_zipfile(download_path):
            extract_zip(download_path, dataset_dir)
        else:
            print(f"Downloaded file is not a ZIP archive: {download_path}")
            print("The file has been saved to the dataset directory.")
        
        print("\nDownload and extraction completed successfully!")
    else:
        print("Download failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()