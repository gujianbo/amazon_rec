
import os
import sys
sys.path.append("..")
from utils.args import config
from tqdm.auto import tqdm
import hashlib
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)


def slice_data(input_file, train_file, test_file, sample_cnt, seed, if_dnn=0):
    # fdout_train = open(train_file+".1.1", "w")
    # fdout_test = open(test_file+".1.1", "w")
    # cnt_train = 0
    # cnt_test = 0
    # train_idx = 1
    # test_idx = 1
    # train_local = "1"
    # test_local = "1"
    fd_dict = dict()
    for local_code in range(6):
        fd_dict['test.'+str(local_code+1)+'.1'] = open(test_file+'.'+str(local_code+1)+'.1', "w")
    for local_code in range(6):
        for i in range(1, sample_cnt):
            fd_dict['train.'+str(local_code+1)+'.'+str(i)] = open(train_file+'.'+str(local_code+1)+'.'+str(i), "w")
    with open(input_file, "r") as fd:
        last_prev_items = ""
        buf = []
        for line in tqdm(fd, ):
            line = line.strip()
            try:
                if if_dnn == 1:
                    (prev_items, candi, candi_id, locale_code, item_feat_str, session_stat_feat_str, interact_feat_str,
                     local_sec_feat_str, locale_code_feat_str, label) = line.split("\t")
                else:
                    (prev_items, candi, locale_code, item_feat_str, session_stat_feat_str, interact_feat_str,
                     local_sec_feat_str, locale_code_feat_str, label) = line.split("\t")
            except:
                print(line)
                exit(1)
            locale_code = locale_code
            if last_prev_items == "" or prev_items == last_prev_items:
                buf.append([line, locale_code])
                last_prev_items = prev_items
                continue
            # 确定是train还是test
            hash_val = int(hashlib.md5((last_prev_items + str(seed)).encode('utf8')).hexdigest()[0:10], 16) % sample_cnt
            for (item, locale_code) in buf:
                if hash_val == 0:
                    key = "test."+locale_code+".1"
                else:
                    key = "train."+locale_code+"."+str(hash_val)
                fd = fd_dict[key]
                fd.write(item+"\n")

                # if hash_val == 0:
                #     if test_local != locale_code:
                #         test_local = locale_code
                #         test_idx = 1
                #         file_name = test_file+"."+test_local+"."+str(test_idx)
                #         fdout_test.close()
                #         fdout_test = open(file_name, "w")
                #         cnt_test = 0
                #     elif cnt_test >= 1000000:
                #         test_idx += 1
                #         file_name = test_file + "." + test_local + "." + str(test_idx)
                #         fdout_test.close()
                #         fdout_test = open(file_name, "w")
                #         cnt_test = 0
                #     fdout_test.write(item+"\n")
                #     cnt_test += 1
                # else:
                #     if train_local != locale_code:
                #         train_local = locale_code
                #         train_idx = 1
                #         file_name = train_file + "." + train_local + "." + str(train_idx)
                #         fdout_train.close()
                #         fdout_train = open(file_name, "w")
                #         cnt_train = 0
                #     elif cnt_train >= 3000000:
                #         train_idx += 1
                #         file_name = train_file + "." + train_local + "." + str(train_idx)
                #         fdout_train.close()
                #         fdout_train = open(file_name, "w")
                #         cnt_train = 0
                #     fdout_train.write(item+"\n")
                #     cnt_train += 1

            buf = []
            buf.append([line, locale_code])
            last_prev_items = prev_items
        hash_val = int(hashlib.md5((last_prev_items + str(seed)).encode('utf8')).hexdigest()[0:10], 16) % sample_cnt
        for (item, locale_code) in buf:
            # if hash_val == 0:
            #     fdout_test.write(item + "\n")
            # else:
            #     fdout_train.write(item + "\n")
            if hash_val == 0:
                key = "test." + locale_code + ".1"
            else:
                key = "train." + locale_code + "." + str(hash_val)
            fd = fd_dict[key]
            fd.write(item + "\n")

        for key in fd_dict:
            fd_dict[key].close()


if __name__ == "__main__":
    logging.info(f"input_file:{config.input_file}")
    logging.info(f"train_file:{config.train_file}")
    logging.info(f"test_file:{config.test_file}")
    logging.info(f"sample_cnt:{config.sample_cnt}")
    logging.info(f"seed:{config.seed}")
    slice_data(config.input_file, config.train_file, config.test_file, config.sample_cnt, config.seed, config.if_dnn)
