# Bank-Transaction-Categorisation
This repository is the code of my dissertation project for MSc Business Analytics, University College London

## Dependencies
-	Scikit-learn version 0.20.3 
-	Spacy version 2.1.4 
-	Gensim version 3.7.3 
-	FastText version 0.9.1 
- xgboost version 0.9
-	Keras version 2.2.4 

## Prereqresites
Download pre-trained word-embeddings files
1. Word2Vec <a href ="https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing">`GoogleNews-vectors-negative300.bin`</a> (from : https://code.google.com/archive/p/word2vec/)
2. FastText <a href ="https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.bin.zip">`wiki-news-300d-1M-subword.bin`</a> (from : https://fasttext.cc/docs/en/english-vectors.html)
3. GloVe <a href="http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip">`glove.42B.300d.zip`</a> and <a href="http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip">`glove.840B.300d.zip`</a> (from : https://github.com/stanfordnlp/GloVe) 
  and pre-process GloVe pre-trained models with `./utils/load_glove.py`
## Running the code
1. Edit `ROOT_PATH`, `DATA_FILES` and pre-trained word-embedding file paths in `./utils/config.py`. 
    Note that the `DATA_FILES` is in a dict object (file_reference: filename.xlsx) For example, 
    ```
     DATA_FILES = {
                    'file1': 'file1.xlsx',
                    'file2': 'file2.xlsx',
                    ...
                  }
    ```
2. Put transaction data files (.xlsx) into folder `./Transaction Data`
3. Run Jupyter Notebook files in `./notebooks/' in order

## Project file structure
```
.
├── notebooks
│   ├── 0_Split_Train_Test_Data.ipynb
│   ├── 1_Select_Text_Representation.ipynb
│   ├── 2_Add_numeric_features_and_model_selection.ipynb
│   ├── 3_Neural_Network.ipynb
│   ├── 4_Predict_Test_Data.ipynb
│   ├── grid_search.ipynb.ipynb
├── output
│   ├── embedding_benchmark_results
│   │   └── <store outputs from 1_Select_Text_Representation.ipynb>
│   ├── embedding_results
│   │   └── <store outputs from 1_Select_Text_Representation.ipynb>
│   ├── final_model
│   │   └── <store outputs from 4_Predict_Test_Data.ipynb>
│   ├── model_results
│   │   └── <store outputs from 2_Add_numeric_features_and_model_selection.ipynb and 3_Neural_Network.ipynb>
│   └── text_embeddings
│       └── <store outputs from 1_Select_Text_Representation.ipynb>
├── resources
│   ├── country_code.csv
│   ├── currency.csv
│   ├── nz_cities.csv
│   ├── nzpostcodes_v2.csv
│   └── stop_words.csv
├── Transaction Data
│   └── <Put transaction files here>
└── utils
    ├── config.py -> Edit config in this file
    ├── feature_eng_utils.py
    ├── load_glove.py
    ├── misc_utils.py
    ├── ml_models.py
    ├── neural_network_model.py
    ├── text_embedding.py
    └── txn_reader.py
 ```
