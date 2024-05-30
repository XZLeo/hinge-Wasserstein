export PYTHONPATH=.
python src/train_nll.py -c /data/datasets/HLW2/metadata.csv \
                -t /data/datasets/HLW2/split/train.txt \
                -v /data/datasets/HLW2/split/val.txt \
                -i /data/datasets/HLW2/images \
                -b ./data/bins.mat \
                --training_name nll_Gaussian \
                --std 4