import os
import shutil
import hashlib
import argparse
from pathlib import Path

def prepare_zenodo_upload(base_dir="NuSHRED_Datasets"):
    """
    Cleans up junk files and individually zips all 'D' dataset directories.
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Directory '{base_path.resolve()}' not found.")
        return

    # 1. Define the cache and system files we want to eradicate
    junk_dirs = {'__pycache__', '.ipynb_checkpoints', '.pytest_cache'}
    junk_files = {'.DS_Store', 'Thumbs.db'}

    # 2. Dynamically find all subdirectories starting with 'D' (D1, D2, ..., Dn)
    dataset_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('D')]
    
    if not dataset_dirs:
        print(f"No dataset directories (starting with 'D') found in {base_path.resolve()}.")
        return

    print(f"Found {len(dataset_dirs)} dataset directories. Starting preparation...\n")

    # Clear previous checksum file if it exists so we don't append indefinitely
    checksum_file = base_path / "checksums.txt"
    if checksum_file.exists():
        checksum_file.unlink()

    for d_path in dataset_dirs:
        print(f"--- Processing {d_path.name} ---")
        
        # 3. Clean up the folder contents
        for root, dirs, files in os.walk(d_path, topdown=False):
            # Remove junk directories
            for j_dir in list(dirs): 
                if j_dir in junk_dirs:
                    target = Path(root) / j_dir
                    print(f"  Removing cache dir: {target}")
                    shutil.rmtree(target)
                    dirs.remove(j_dir) # Prevent os.walk from entering a deleted dir
            
            # Remove junk files and compiled python files
            for file in files:
                if file in junk_files or file.endswith('.pyc'):
                    target = Path(root) / file
                    print(f"  Removing junk file: {target}")
                    target.unlink()

        # 4. Zip the directory
        # This zips 'D1' so that when extracted, it creates a 'D1' folder, not a mess of files
        zip_target_path = base_path / d_path.name
        print(f"  Zipping {d_path.name}...")
        shutil.make_archive(
            base_name=str(zip_target_path), 
            format='zip', 
            root_dir=base_path, 
            base_dir=d_path.name
        )
        
        # 5. Generate MD5 Checksum
        zip_file = zip_target_path.with_suffix('.zip')
        md5_hash = hashlib.md5()
        
        with open(zip_file, "rb") as f:
            # Read in 4K chunks to prevent memory overload on huge zips
            for byte_block in iter(lambda: f.read(4096), b""):
                md5_hash.update(byte_block)
        
        checksum = md5_hash.hexdigest()
        print(f"  MD5 Checksum: {checksum}")
        
        # Save checksum to the master text file
        with open(checksum_file, "a") as f:
            f.write(f"{checksum}  {zip_file.name}\n")

    print("\nAll datasets cleaned and zipped successfully!")
    print(f"A master checksum file has been saved to: {checksum_file}")

if __name__ == "__main__":
    # Get the directory where this script actually lives
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the default path as one level up from the script
    default_dir = os.path.join(script_dir, "..", "NuSHRED_Datasets")

    parser = argparse.ArgumentParser(description="Clean and zip dataset folders for Zenodo upload.")
    
    # --dir (or -d): Defaults to the parent directory's NuSHRED_Datasets folder
    parser.add_argument(
        "-d", "--dir", 
        type=str, 
        default=default_dir, 
        help="Target base directory containing datasets (default: ../NuSHRED_Datasets)"
    )

    args = parser.parse_args()

    prepare_zenodo_upload(base_dir=args.dir)