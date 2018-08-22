nohup python test/qa_test/disan_train.py  --word2vec_file /DeepLearning/ray_li/glove.6B.300d.txt --train_data test/qa_test/data/train_data.csv --validate_data test/qa_test/data/valid_data.csv >/dev/null 2>qa_disan_log &
nohup python test/qa_test/siamese_train.py --word2vec_model /DeepLearning/ray_li/glove.6B.300d.txt --training_data train_data.csv --validate_data valid_data.csv  >/dev/null 2>log.2 &

bimpm qa train:
python data_test/qa_test/bimpm_train.py --learning_rate 0.0005 --highway_layer_num 1 --train_data data_test/qa_test/data/train_data.csv --validate_data data_test/qa_test/data/valid_data.csv --test_data data_test/qa_test/data/test_data.csv  --word2vec_file data_test/qa_test/data/glove.6B.300d.txt --gpu 0
bimpm qa test:
python data_test/qa_test/bimpm_predict.py --test_data data_test/qa_test/data/test_data.csv --gpu 1 --model_directory quora_disan_runs/1528356750/ --word2vec_file data_test/qa_test/data/glove.6B.300d.txt
