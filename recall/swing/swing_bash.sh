#!/bin/sh

root=$1
echo ${root}
raw_folder=$2
echo ${raw_folder}
out_folder=$3
echo ${out_folder}


# 生成item pair
nohup python -u swing_gen_item_pair.py --input_file ${root}/data/sessions_train.csv \
  --output_file ${root}/${out_folder}/swing.sim \
  --log_file ${root}/log/swing_pair.log > log 2>&1 &

# 对item pair进行排序后uniq，删掉只有一个的pair
for i in $(seq 0 9)
do
  for j in $(seq 0 9)
  do
    echo "process ${root}/${out_folder}/pair/pair_${i}_${j}"
    sort ${root}/${out_folder}/pair/pair_${i}_${j} | uniq -c | awk '{if($1!="1") print $2}' > ${root}/${out_folder}/pair/uniq_pair_${i}_${j}
  done
done

# 根据uniq后的pair进行相似度计算
nohup python -u swing_sim_calc.py --output_file ${root}/${out_folder}/swing.sim \
  --log_file ${root}/log/swing_sim.log > log 2>&1 &

# 对计算的相似度进行过滤，保留每个topk
nohup python -u swing_sim_uniq.py --output_file ${root}/${out_folder}/swing.sim \
  --log_file ${root}/log/swing_sim.log > log 2>&1 &



python -u swing_sim_uniq.py --output_file ${root}/${out_folder}/swing.sim \
  --log_file ${root}/log/swing_sim.log
