import os
import subprocess
import argparse


class WikiDownloader:
    def __init__(self, language, save_path):
        self.save_path = save_path + '/wikicorpus_' + language

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.language = language
        # Use a mirror from https://dumps.wikimedia.org/mirrors.html if the below links do not work
        self.download_urls = {
            # 'en' : 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2',
            'en' : 'https://mirror.accum.se/mirror/wikimedia.org/dumps/enwiki/20230101/enwiki-20230101-pages-articles.xml.bz2',
            'zh' : 'https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2'
        }

        self.output_files = {
            'en' : 'wikicorpus_en.xml.bz2',
            'zh' : 'wikicorpus_zh.xml.bz2'
        }

    def download(self):
        if self.language in self.download_urls:
            url = self.download_urls[self.language]
            filename = self.output_files[self.language]

            print('Downloading:', url)
            if os.path.isfile(self.save_path + '/' + filename):
                print('** Download file already exists, skipping download')
            else:
                cmd = ['wget', url, '--output-document={}'.format(self.save_path + '/' + filename)]
                print('Running:', cmd)
                status = subprocess.run(cmd)
                if status.returncode != 0:
                    raise RuntimeError('Wiki download not successful')

            # Always unzipping since this is relatively fast and will overwrite
            print('Unzipping:', self.output_files[self.language])
            subprocess.run('bzip2 -dk ' + self.save_path + '/' + filename, shell=True, check=True)

        else:
            assert False, 'WikiDownloader not implemented for this language yet.'


def parse_args():
    parser = argparse.ArgumentParser(description="wiki downloader")
    parser.add_argument("--language", type=str, default='en', help="en or zh")
    parser.add_argument("--save_path", type=str, default='./', help="save_path/wikicorpus_language")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    downloader = WikiDownloader(args.language, args.save_path)
    downloader.download()
