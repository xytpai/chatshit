#### 1. Download wiki dataset for pretraining

```bash
rm -rf ./wiki/ # Change to your own path
python ./data/wiki_downloader.py --language=en --save_path=./wiki/
# Or download by chrome then execute: bzip2 -dk ${bz2file}
```

#### 2. Extract wiki dataset for pretraining

```bash
python ./data/WikiExtractor.py ${xmlfile} -o ${outputdir}
```

#### 3. Cleanup wiki dataset for pretraining

```bash
cd data/wikicleaner/
bash run.sh '/data/wiki/text/*/wiki_??' /data/wiki/results # Change to your own path
```
