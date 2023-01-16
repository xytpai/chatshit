#### 1. Download wiki dataset for pretraining

```bash
rm -rf ./wiki/
python data/wiki_downloader.py --language=en --save_path=./wiki/
# Or download by chrome then execute: bzip2 -dk ${bz2file}
```

#### 2. Extract wiki dataset for pretraining

```bash
python ./data/wikiextractor/WikiExtractor.py ${xmlfile} -o ${outputdir}
```

#### 3. Cleanup wiki dataset for pretraining

```bash
```
