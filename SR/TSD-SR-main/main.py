import argparse
import os
from html.parser import HTMLParser
from urllib.parse import urljoin

import requests


ROOT_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/"
DEFAULT_PATTERN = "DIV2K_train_HR.zip"
CHUNK_SIZE = 1024 * 1024


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return

        for key, value in attrs:
            if key == "href" and value:
                self.links.append(value)


def fetch_links(root_url):
    response = requests.get(root_url, timeout=30)
    response.raise_for_status()

    parser = LinkParser()
    parser.feed(response.text)
    return [urljoin(root_url, link) for link in parser.links]


def select_files(links, pattern):
    return [
        link for link in links
        if pattern in os.path.basename(link)
    ]


def download_file(url, output_dir):
    file_name = os.path.basename(url)
    output_path = os.path.join(output_dir, file_name)

    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with open(output_path, "wb") as file_obj:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    file_obj.write(chunk)

    return output_path


def main(root_url, output_dir, pattern):
    os.makedirs(output_dir, exist_ok=True)

    links = fetch_links(root_url)
    matched_files = select_files(links, pattern)

    if not matched_files:
        raise ValueError(
            f"No files matched pattern '{pattern}' at {root_url}"
        )

    print(f"Found {len(matched_files)} file(s) matching '{pattern}'")

    for index, url in enumerate(matched_files, start=1):
        print(f"[{index}/{len(matched_files)}] Downloading {url}")
        output_path = download_file(url, output_dir)
        print(f"Saved to {output_path}")

    print("Download completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download files from the DIV2K index by filename pattern."
    )
    parser.add_argument(
        "output_dir",
        help="Directory where the downloaded files will be saved.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Substring to match in filenames. Default: {DEFAULT_PATTERN}",
    )
    parser.add_argument(
        "--root-url",
        default=ROOT_URL,
        help=f"Dataset index URL. Default: {ROOT_URL}",
    )
    args = parser.parse_args()

    main(args.root_url, args.output_dir, args.pattern)
