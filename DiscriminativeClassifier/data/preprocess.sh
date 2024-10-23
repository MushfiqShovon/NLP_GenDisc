#!/bin/sh

cd $1

# shuffle the dataset
shuf -o test.csv < test.csv
shuf -o train.csv < train.csv

# random sample 5000 from train.csv.shuf as test.csv.shuf
head -n 5000 train.csv > valid.csv
tail -n 555000 train.csv > train.csv.bak
mv train.csv.bak train.csv

# text format in *.csv : label \t title \t content \m
# get_content.py selects the content column.
python ../get_content.py

# tokenize the file with tokenizer-noescape.perl
sed -i 's/\\n/\ /g' train_content.txt
sed -i 's/\\"/"/g' train_content.txt
sed -i 's/\\n/\ /g' valid_content.txt
sed -i 's/\\"/"/g' valid_content.txt
sed -i 's/\\n/\ /g' test_content.txt
sed -i 's/\\"/"/g' test_content.txt
perl ../tokenizer-noescape.perl -l en < train_content.txt > train_content.txt.token
perl ../tokenizer-noescape.perl -l en < valid_content.txt > valid_content.txt.token
perl ../tokenizer-noescape.perl -l en < test_content.txt > test_content.txt.token

# generate train/valid/test.data file. Format: label \t tokenized_content
python ../gen_format_file.py

rm *_content.txt
rm *_content.txt.token
# rm *.csv
mkdir data
mv *.data data/

# sample a subset of instances: for data efficiency setting
# sample #SAMPLE instances per class
for i in 5000
do
        python ../sample.py --sample_num ${i}
done

# convert text to token id
max_len=80 # 200
python ../preprocess_data.py --vocab_size 40000 --max_len ${max_len} --num_instance 5000
# python ../preprocess_data.py --vocab_size 40000 --max_len ${max_len}  
