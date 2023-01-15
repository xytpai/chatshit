import os
import glob
import argparse
import subprocess


class WikicorpusTextFormatting:
    def __init__(self, wiki_path, output_filename, recursive=False):
        self.wiki_path = wiki_path
        self.recursive = recursive
        self.output_filename = output_filename

    # This puts one article per line
    def merge(self):
        with open(self.output_filename, mode='w', newline='\n') as ofile:
            for dirname in glob.glob(self.wiki_path + '/*/', recursive=False):
                for filename in glob.glob(
                        dirname + 'wiki_*', recursive=self.recursive):
                    print(filename)
                    article_lines = []
                    article_open = False

                    with open(filename, mode='r', newline='\n') as file:
                        for line in file:
                            if '<doc id=' in line:
                                article_open = True
                            elif '</doc>' in line:
                                article_open = False
                                for oline in article_lines[1:]:
                                    if oline != '\n':
                                        ofile.write(oline.rstrip() + " ")
                                ofile.write("\n\n")
                                article_lines = []
                            else:
                                if article_open:
                                    article_lines.append(line)


def parse_args():
    parser = argparse.ArgumentParser(description="wiki prep")
    parser.add_argument("--p", type=str, default='./wikiextractor/WikiExtractor.py')
    parser.add_argument("--input", type=str, default='./wikicorpus_en/wikicorpus_en.xml')
    parser.add_argument("--n_processes", type=str, default='16')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    wiki_dir = os.path.dirname(args.input)
    wiki_type = os.path.basename(args.input).split('.')[0].strip()

    wikiextractor_command = 'python ' + args.p + ' ' \
        + args.input + ' -b 100M --processes ' \
            + args.n_processes + ' -o ' + wiki_dir + '/json --json'
    print('WikiExtractor Command:', wikiextractor_command)
    
    wikiextractor_process = subprocess.run(wikiextractor_command, shell=True, check=True)
    
    output_file_name = wiki_dir + '/' + wiki_type + '_one_article_per_line.txt'
    wiki_formatter = WikicorpusTextFormatting(wiki_dir, output_file_name, recursive=True)
    wiki_formatter.merge()

    assert os.stat(output_file_name).st_size > 0, 'File glob did not pick up extracted wiki files from WikiExtractor.'


if __name__ == '__main__':
    main()
