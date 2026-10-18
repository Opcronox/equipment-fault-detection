"""Download the official UCI AI4I 2020 predictive maintenance CSV (CC BY 4.0)."""
from __future__ import annotations

import argparse
import io
from pathlib import Path
import zipfile

import requests

ROOT = Path(__file__).resolve().parent
DEFAULT_DEST = ROOT / 'data' / 'ai4i2020.csv'
DATASET_PAGE = 'https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset'
UCI_ZIP_URLS = [
    'https://archive.ics.uci.edu/static/public/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip',
    'https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip',
]


def download(destination: Path = DEFAULT_DEST, force: bool = False) -> Path:
    destination = Path(destination)
    if destination.is_file() and not force:
        print(f'Dataset already exists: {destination}')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    last_error = None
    for url in UCI_ZIP_URLS:
        try:
            response = requests.get(url, timeout=45, headers={'User-Agent': 'Educational-ML-Project/1.0'})
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                matches = [n for n in archive.namelist() if n.lower().endswith('ai4i2020.csv')]
                if not matches:
                    raise RuntimeError('Archive has no ai4i2020.csv file')
                with archive.open(matches[0]) as source:
                    csv_bytes = source.read()
                if b'Machine failure' not in csv_bytes[:1500]:
                    raise RuntimeError('Downloaded CSV header did not match the expected dataset')
                destination.write_bytes(csv_bytes)
            print(f'Downloaded {destination} ({len(csv_bytes):,} bytes)')
            print(f'Please attribute the UCI AI4I 2020 dataset: {DATASET_PAGE}')
            return destination
        except (requests.RequestException, zipfile.BadZipFile, RuntimeError) as exc:
            last_error = exc

    raise RuntimeError(
        'Could not automatically download the dataset. Visit\n'
        f'  {DATASET_PAGE}\n'
        'Click Download, extract ai4i2020.csv, and save it at\n'
        f'  {destination}\n'
        f'Last download error: {last_error}'
    ) from last_error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true', help='Download again even if a CSV exists')
    args = parser.parse_args()
    download(force=args.force)
