#### 1. Download wiki dataset for pretraining

```bash
rm -rf /data/wiki/ # Change to your own path
python ./data/wiki_downloader.py --language=en --save_path=/data/wiki/
# Or download by chrome then execute: bzip2 -dk ${bz2file}
```

#### 2. Extract wiki dataset for pretraining

```bash
python ./data/WikiExtractor.py /data/wiki/enwiki-20230101-pages-articles.xml -o /data/wiki/text
```

#### 3. Cleanup wiki dataset for pretraining

```bash
cd data/wikicleaner/
bash run.sh '/data/wiki/text/*/wiki_??' /data/wiki/results # Change to your own path
```

#### 4. Download vocab

```bash
cd .. # To ./data
python vocab_downloader.py --type=bert-base-uncased
mv bert-base-uncased-vocab.txt /data/wiki/
```

#### 5. Create bert pretraining dataset

```bash
export VOCAB_FILE=/data/wiki/bert-base-uncased-vocab.txt
bash parallel_create_pretraining_data.sh /data/wiki/results/
```
