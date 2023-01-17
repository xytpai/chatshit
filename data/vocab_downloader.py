import argparse
import subprocess
import tokenization


def parse_args():
    parser = argparse.ArgumentParser(description="vocab downloader")
    parser.add_argument("--type", type=str, required=True, 
        choices=tokenization.PRETRAINED_VOCAB_ARCHIVE_MAP.keys())
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    url = tokenization.PRETRAINED_VOCAB_ARCHIVE_MAP[args.type]
    subprocess.run(['wget', url])
