import lightgbm as lgb


if __name__ == "__main__":
    logging.info(f"input_file:{config.input_file}")
    logging.info(f"output_file:{config.output_file}")
    logging.info(f"root_path:{config.root_path}")
    item_dict = load_item_feat(config.item_feat_file)
    i2i_dicts = load_i2i_dicts(config.root_path)
    feat_extract(config.input_file, config.output_file, item_dict, i2i_dicts)