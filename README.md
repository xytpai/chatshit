#### 1. Download wiki dataset for pretraining

```bash
rm -rf ./wiki/
python data/wiki_downloader.py --language=en --save_path=./wiki/
```

#### 2. Pre-processing wiki dataset for pretraining

```bash
python data/wiki_prep.py \
--p=./data/wikiextractor/WikiExtractor.py \
--input=./wiki/wikicorpus_en/wikicorpus_en.xml \
--n_processes=32
```
