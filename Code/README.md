Every paper has its own directory, containing a set of notebooks aimed at reproducing the results of the paper.

In addition, there are two scripts in this directory:
- `prepare_datasets4upload.py`: This script prepares the datasets for upload to Zenodo, by zipping each dataset. To execute it, run the following command:
```bash
python prepare_datasets4upload.py
```
- `download_datasets.py`: This script downloads the datasets from Zenodo link, unzips them, and organizes them in the appropriate directories for use in the notebooks. It is possible to download all datasets or only a specific one by providing the dataset name as an argument. To execute it (as an example for `D1` and `D2` datasets), run the following command:
```bash
python download_datasets.py --files D1 D2
```
