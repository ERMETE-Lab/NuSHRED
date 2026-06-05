import os
import requests
import zipfile
import argparse

def download_specific_zenodo_files(record_id, output_dir, files_to_download=None):
    """
    Downloads specific files from a Zenodo record.
    If files_to_download is None or empty, it downloads everything.
    Automatically deletes .zip files after successful extraction.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Automatically append '.zip' to any file that doesn't have it
    if files_to_download:
        files_to_download = [f if f.endswith('.zip') else f"{f}.zip" for f in files_to_download]
        print(f"Targeting specific files: {files_to_download}")
    
    api_url = f"https://zenodo.org/api/records/{record_id}"
    print(f"Fetching metadata for Zenodo record {record_id}...")
    
    response = requests.get(api_url)
    response.raise_for_status()
    
    record_data = response.json()
    all_files = record_data.get('files', [])
    
    if not all_files:
        print("No files found in this Zenodo record.")
        return

    # Filter the files if a specific list was provided
    if files_to_download:
        files = [f for f in all_files if f.get('key') in files_to_download]
        if not files:
            print(f"None of the requested files {files_to_download} were found in this record.")
            print("Available files are:", [f.get('key') for f in all_files])
            return
    else:
        files = all_files

    print(f"Found {len(files)} matching file(s). Starting download...")
    
    for file_info in files:
        file_name = file_info.get('key')
        download_url = file_info.get('links', {}).get('self')
        
        file_path = os.path.join(output_dir, file_name)
        print(f"Downloading {file_name}...")
        
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
        print(f"Saved: {file_path}")
        
        if file_name.endswith('.zip'):
            print(f"Extracting {file_name}...")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            print(f"Extracted {file_name} successfully.")

            # ALWAYS delete the zip file after successful extraction
            try:
                os.remove(file_path)
                print(f"Deleted zip file: {file_name}")
            except OSError as e:
                print(f"Error deleting zip file {file_name}: {e}")
            
    print("\nProcess completed!")

if __name__ == "__main__":
    # Get the directory where this script actually lives
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the default path as one level up from the script
    default_out = os.path.join(script_dir, "..", "NuSHRED_Datasets")

    parser = argparse.ArgumentParser(description="Download specific dataset files from Zenodo.")
    
    parser.add_argument(
        "-f", "--files", 
        nargs="*", 
        default=None, 
        help="List of specific datasets to download (e.g., D1 D2). If omitted, downloads all."
    )
    
    parser.add_argument(
        "-r", "--record", 
        type=str, 
        default="20554287", 
        help="Zenodo record ID (default: 20554287)"
    )
    
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default=default_out, 
        help="Target output folder (default: parent directory of script / NuSHRED_Datasets)"
    )

    args = parser.parse_args()

    download_specific_zenodo_files(
        record_id=args.record, 
        output_dir=args.output, 
        files_to_download=args.files
    )