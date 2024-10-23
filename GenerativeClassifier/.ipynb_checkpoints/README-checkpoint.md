# Generative Text Classifier
This repository contains code in Pytorch for generative text classifier. It can be used to reproduce the results in the following article:

Xiaoan Ding, Kevin Gimpel. [Latent-Variable Generative Models for Data-Efficient Text Classification](https://arxiv.org/abs/1910.00382). 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP 2019)


## dataset
We present our results on six publicly available text classification datasets introduced by Zhang et al. (2015). 

The datasets are available at http://goo.gl/JyCnZq.

## data preprocessing
We take dataset dbpedia as an example. Download the dbpedia_csv.tar.gz under data directory.

```sh
$ cd data
$ tar -xvf dbpedia_csv.tar
$ mv dbpedia_csv dbpedia
$ sh preprocess.sh dbpedia
```

## generative model
File: gen_train.py

Run:
```sh
python gen_train.py --data DATASET--word_emb_dim WORD_EMBED_DIM --hid_dim HIDDEN_DIM --label_emb_dim LABEL_DIM --epochs NUM_EPOCHS --batch_size BATCH_SIZE --log_interval N --save_dir SAVE_DIR --cuda --dropout DR --lr LR  
```

Example:
```sh
python gen_train.py --cuda --save_prefix gen --dataset ag_news --datafile traindata.v40000.l80.s20 --epochs 100
```

## latent generative model
File: latent_gen_train.py

Run:
```sh
python latent_gen_train.py --mode LATENT_MODEL_TYPE --data DATASET --datafile TOKENIZED_FILE --word_emb_dim WORD_EMBED_DIM --hid_dim HIDDEN_DIM --cond_emb_dim LATENT_EMB_DIM --label_emb_dim LABEL_DIM  --ncondition NUM_LATENT --epochs N_EPOCHS --batch_size BATCH_SIZE --log_interval N --save_interval N --cuda --dropout DR --lr LR 
```

Example:
```sh
python latent_gen_train.py --cuda --mode auxiliary --save_prefix aux_gen --dataset dbpedia --datafile traindata.v40000.l80.s5 --save_interval 10 --epochs 100
or

python latent_gen_train.py --cuda --mode auxiliary --save_prefix aux_gen --dataset dbpedia --datafile traindata.v40000.l80.s5 --save_interval 10 --epochs 100 --resume aux_gen_best.chkp
```
