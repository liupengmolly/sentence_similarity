disan with pinyin train
nohup python data_test/pai/disan_train.py --word2vec_file data_test/pai/data/word_char_embed.json --train_data data_test/pai/data/train_data.csv --validate_data data_test/pai/data/valid_data.csv --max_sentence_length 80 --feature_type word+char --dropout 0.95 --batch_size 32 --model_save_path disan --log_name disan --gpu 1 > /dev/null 2> disan_print &
disan with pinyin predict:
python data_test/pai/disan_predict.py --word2vec_file data_test/pai/data/atec_w2v_with_pinyin.json --feature_type char --use_pinyin True --batch_size 50 --max_sentence_length 70 --model_directory pai_disan_runs/pinyin_train0 --test_data data_test/pai/data/test_data.csv --gpu 0

fast_disan with pinyin train
nohup python data_test/pai/fast_disan_train.py --word2vec_file data_test/pai/data/word_char_embed.json --train_data data_test/pai/data/train_data.csv --validate_data data_test/pai/data/valid_data.csv --max_sentence_length 80 --feature_type word+char --dropout 0.95 --model_save_path fastdisan_0.95 --log_name 0.95 --gpu 0 > /dev/null 2> fastdisan_print_0.95 &
fast_disan with pinyin predict(flexible arguments : word_embedding length)
python data_test/pai/fast_disan_predict.py --word2vec_file data_test/pai/data/atec_w2v_with_pinyin.json --feature_type char --use_pinyin True --dropout 0.95 --max_sentence_length 70 --model_directory pai_fast_disan_runs/py_train01_0.95_new --test_data data_test/pai/data/test_data.csv --gpu 5 > /dev/null 2> print_0.95 &

python data_test/pai/bimpm_train.py --word2vec_file data_test/pai/data/word_char_embed.json --learning_rate 0.0005 --highway_layer_num 1 --train_data data_test/pai/data/train_data.csv --validate_data data_test/pai/data/valid_data.csv --max_sentence_length 80 --feature_type word+char --model_save_path bimpm --log_name --gpu 2 > /dev/null 2> bimpm_print &

python data_test/pai/bimpm_predict.py --test_data data_test/pai/data/test_data.csv --model_directory  pai_bimpm_runs/bimpm --gpu 4

python data_test/pai/disan_bimpm_ensemble.py --train_data data_test/pai/data/train_data2.csv --validate_data data_test/pai/data/test_data.csv --test_data data_test/pai/data/predict_data.csv --gpu 0

Matchpyramid 训练:
python data_test/pai/matchpyramid.py --phase train --model_file data_test/pai/matchpyramid.config

feature_ensemble 训练
python data_test/pai/feature_ensemble.py --train_data data_test/pai/data/train_data2.csv --validate_data data_test/pai/data/balance_data/test_data.csv --test_data data_test/pai/data/predict_data.csv --gpu 0

