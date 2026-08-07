import os
import requests
import zipfile
import argparse

# Concept record ID — always resolves to the latest Zenodo version (DOI 10.5281/zenodo.13789584)
DEFAULT_ZENODO_RECORD = "13789584"

# Legacy D1–D5 names from the first Zenodo release
LEGACY_ALIASES = {
    "D1": "MSFR",
    "D2": "MSFR",
    "D3": "DYNASTY",
    "D4": "TRIGA",
    "D5": "LRA-neutronics",
}

# Zip keys to try on Zenodo (new names first, then legacy archives)
ZIP_CANDIDATES = {
    "MSFR": ["MSFR.zip", "D1.zip", "D2.zip"],
    "DYNASTY": ["DYNASTY.zip", "D3.zip"],
    "TRIGA": ["TRIGA.zip", "D4.zip"],
    "LRA-neutronics": ["LRA-neutronics.zip", "D5.zip"],
    "RDA": ["RDA.zip"],
}


def normalize_dataset_names(names):
    """Map legacy aliases (D1, …) to canonical dataset names; deduplicate."""
    canonical = []
    for name in names:
        key = LEGACY_ALIASES.get(name.upper(), name)
        if key not in canonical:
            canonical.append(key)
    return canonical


def resolve_zip_keys(requested_names, available_keys):
    """
    Expand canonical dataset names to zip file keys present in the Zenodo record.
    For MSFR on legacy uploads, both D1.zip and D2.zip are fetched when present.
    """
    available = set(available_keys)
    zip_keys = []

    for dataset in normalize_dataset_names(requested_names):
        candidates = ZIP_CANDIDATES.get(dataset, [f"{dataset}.zip"])
        matched = [name for name in candidates if name in available]
        if not matched:
            print(
                f"Warning: no archive found for '{dataset}'. "
                f"Tried: {candidates}. Available: {sorted(available_keys)}"
            )
            continue
        for name in matched:
            if name not in zip_keys:
                zip_keys.append(name)

    return zip_keys


def download_specific_zenodo_files(record_id, output_dir, files_to_download=None):
    """
    Downloads specific files from a Zenodo record.
    If files_to_download is None or empty, it downloads everything.
    Automatically deletes .zip files after successful extraction.
    """
    os.makedirs(output_dir, exist_ok=True)

    api_url = f"https://zenodo.org/api/records/{record_id}"
    print(f"Fetching metadata for Zenodo record {record_id}...")

    response = requests.get(api_url, allow_redirects=True)
    response.raise_for_status()

    record_data = response.json()
    resolved_id = record_data.get("id", record_id)
    if str(resolved_id) != str(record_id):
        print(f"Resolved concept record {record_id} → version {resolved_id}")

    all_files = record_data.get("files", [])

    if not all_files:
        print("No files found in this Zenodo record.")
        return

    available_keys = [f.get("key") for f in all_files]

    if files_to_download:
        zip_keys = resolve_zip_keys(files_to_download, available_keys)
        if not zip_keys:
            print("No matching files to download.")
            return
        print(f"Targeting archives: {zip_keys}")
        files = [f for f in all_files if f.get("key") in zip_keys]
    else:
        files = all_files

    print(f"Found {len(files)} matching file(s). Starting download...")

    for file_info in files:
        file_name = file_info.get("key")
        download_url = file_info.get("links", {}).get("self")

        file_path = os.path.join(output_dir, file_name)
        print(f"Downloading {file_name}...")

        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"Saved: {file_path}")

        if file_name.endswith(".zip"):
            print(f"Extracting {file_name}...")
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)

            print(f"Extracted {file_name} successfully.")

            try:
                os.remove(file_path)
                print(f"Deleted zip file: {file_name}")
            except OSError as e:
                print(f"Error deleting zip file {file_name}: {e}")

    print("\nProcess completed!")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_out = os.environ.get(
        "NUSHRED_DATA_DIR", os.path.join(script_dir, "..", "NuSHRED_Datasets")
    )

    parser = argparse.ArgumentParser(
        description="Download dataset archives from Zenodo.",
        epilog=(
            "Legacy names D1–D5 are accepted (e.g. D1 and D2 both map to MSFR). "
            f"Default record is the concept DOI {DEFAULT_ZENODO_RECORD} (10.5281/zenodo.13789584)."
        ),
    )

    parser.add_argument(
        "-f",
        "--files",
        nargs="*",
        default=None,
        help=(
            "Datasets to download (e.g. MSFR DYNASTY). "
            "Legacy aliases D1–D5 are supported. If omitted, downloads all files."
        ),
    )

    parser.add_argument(
        "-r",
        "--record",
        type=str,
        default=DEFAULT_ZENODO_RECORD,
        help=f"Zenodo record ID (default: {DEFAULT_ZENODO_RECORD}, concept DOI — latest version)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=default_out,
        help="Target output folder (default: $NUSHRED_DATA_DIR, or NuSHRED_Datasets/ at repo root)",
    )

    args = parser.parse_args()

    download_specific_zenodo_files(
        record_id=args.record,
        output_dir=args.output,
        files_to_download=args.files,
    )
