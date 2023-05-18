#!/bin/sh

root=$1
echo ${root}
country=$2
echo ${country}

nohup python -u faiss_i2i.py --input_file ${root}/n2v/n2v_emb.txt --product_file ${root}/data/products_train.csv \
  --product_dict_file ${root}/data/product2id.dict --idproduct_dict_file ${root}/data/id2product.dict \
  --output_file ${root}/n2v/i2i_${country}.txt --country ${country} > log 2>&1 &