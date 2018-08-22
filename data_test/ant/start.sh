#!/usr/bin/env bash
nohup python data_test/ant/disan_train.py --word2vec_file /DeepLearning/ray_li/wikiword2vec.json --train_data data_test/ant/data/train_data.csv --validate_data data_test/ant/data/valid_data.csv --max_sentence_length 70 > /dev/null 2> log_ant_disan &
python data_test/ant/disan_predict.py --word2vec_file /home/ray_li/python27/data_test/ant/data/ant_data_w2v.json --test_data data_test/ant/data/test_data.csv --max_sentence_length 50 --model_directory disan_models/ --gpu 5


nohup python data_test/ant/siamese_train.py --word2vec_model /DeepLearning/ray_li/wikiword2vec.json --training_data train_data.csv --validate_data valid_data.csv --is_char_based False >/dev/null 2>log.2 &
python data_test/ant/siamese_predict.py --word2vec_file /DeepLearning/ray_li/wikiword2vec.json --data_test_data data_test/ant/data/data_test_data.csv  --model_directory ant_disan_runs/1526903801/checkpoints


disan with pinyin train
nohup python data_test/ant/disan_train.py --word2vec_file data_test/ant/data/atec_w2v_with_pinyin.json --train_data data_test/ant/data/train_data0.csv --validate_data data_test/ant/data/valid_data.csv --max_sentence_length 70 --use_pinyin True --feature_type char --model_save_path py_train0 --log_name _py0 --batch_size 36 --gpu 0 > /dev/null 2> log_ant_disan_py0 &
disan with pinyin predict:
python data_test/ant/disan_predict.py --word2vec_file data_test/ant/data/atec_w2v_with_pinyin.json --feature_type char --use_pinyin True --batch_size 50 --max_sentence_length 70 --model_directory ant_disan_runs/pinyin_train0 --test_data data_test/ant/data/test_data.csv --gpu 0


fast_disan with pinyin train
nohup python data_test/ant/fast_disan_train.py --word2vec_file data_test/ant/data/atec_w2v_with_pinyin.json --train_data data_test/ant/data/train_data0.csv --validate_data data_test/ant/data/valid_data.csv --max_sentence_length 70 --use_pinyin True --feature_type char --dropout 0.95 --word_embedding_length 100 --model_save_path py_train0_0.95 --log_name _py0 --gpu 3 > /dev/null 2> log_ant_fastdisan_py0 &
fast_disan with pinyin predict(flexible arguments : word_embedding length)
python data_test/ant/fast_disan_predict.py --word2vec_file data_test/ant/data/atec_w2v_with_pinyin.json --feature_type char --use_pinyin True --dropout 0.95 --max_sentence_length 70 --model_directory ant_fast_disan_runs/py_train01_0.95_new --test_data data_test/ant/data/test_data.csv --gpu 5

stacking_fast_disan with pinyin train
nohup python data_test/ant/fast_disan_train.py --word2vec_file data_test/ant/data/atec_w2v_with_pinyin.json --train_data data_test/ant/data/train_data0.csv --validate_data data_test/ant/data/valid_data.csv --max_sentence_length 70 --use_pinyin True --feature_type char --dropout 0.95 --use_stacking True --word_embedding_length 100 --model_save_path py_train0_0.95_stacking --log_name _py0_stacking --gpu 3 > /dev/null 2> log_ant_fastdisan_py0_stacking &

python data_test/ant/bimpm_train.py --learning_rate 0.0005 --highway_layer_num 1 --train_data data_test/ant/data/train_data0.csv --validate_data data_test/ant/data/valid_data.csv --model_save_path 0 --log_name 0 --gpu 0 &
python data_test/ant/bimpm_predict.py --test_data data_test/ant/data/test_data.csv --model_directory  ant_bimpm_runs/model_3 --gpu 4
python data_test/ant/bimpm_predict.py --test_data data_test/ant/data/test_data.csv --model_directory  ant_bimpm_runs/model_0 --use_pinyin True  --feature_type char --max_sentence_length 70  --word2vec_file data_test/ant/data/atec_w2v_with_pinyin_60d.json    --gpu 2

python data_test/ant/disan_bimpm_ensemble.py --train_data data_test/ant/data/train_data10.csv --test_data data_test/ant/data/test_data.csv --gpu 4
