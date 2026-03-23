import argparse
import urllib.request
import os

def download_data(url, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"Downloading dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Dataset successfully downloaded to: {output_path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset script without hardcoded paths.")
    parser.add_argument("--url", required=True, help="The URL to download the dataset from.")
    parser.add_argument("--output", required=True, help="The local file path to save the dataset.")
    args = parser.parse_args()

    download_data(args.url, args.output)