#!/usr/bin/env python3
"""
Example script showing how to use the download_dataset.py script programmatically.
"""

import subprocess
import sys
from pathlib import Path

def download_dataset(url_or_file_id, filename=None):
    """
    Download a dataset from Google Drive using the download_dataset.py script.
    
    Args:
        url_or_file_id (str): Google Drive URL or file ID
        filename (str, optional): Custom filename for the download
    
    Returns:
        bool: True if successful, False otherwise
    """
    script_path = Path(__file__).parent / "download_dataset.py"
    
    # Build command
    cmd = [sys.executable, str(script_path), url_or_file_id]
    if filename:
        cmd.append(filename)
    
    try:
        # Run the download script
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def main():
    """Example usage of the download function."""
    print("Dataset Download Example")
    print("=" * 50)
    
    # Example 1: Download with URL
    print("\nExample 1: Download from Google Drive URL")
    example_url = "https://drive.google.com/file/d/YOUR_FILE_ID_HERE/view"
    print(f"Usage: download_dataset('{example_url}')")
    
    # Example 2: Download with file ID and custom filename
    print("\nExample 2: Download with file ID and custom filename")
    example_file_id = "YOUR_FILE_ID_HERE"
    example_filename = "my_dataset.zip"
    print(f"Usage: download_dataset('{example_file_id}', '{example_filename}')")
    
    print("\nReplace 'YOUR_FILE_ID_HERE' with your actual Google Drive file ID.")
    print("\nTo use this programmatically, uncomment the lines below:")
    print("# success = download_dataset('YOUR_ACTUAL_FILE_ID', 'dataset.zip')")
    print("# if success:")
    print("#     print('Dataset downloaded successfully!')")
    print("# else:")
    print("#     print('Failed to download dataset.')")

if __name__ == "__main__":
    main()