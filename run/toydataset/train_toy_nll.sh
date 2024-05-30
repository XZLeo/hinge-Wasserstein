export PYTHONPATH=.
python src/train_nll.py -c toydataset/aleatoric/toydata.csv\
                -i toydataset/aleatoric/images\
                -t toydataset/aleatoric/split/train.txt\
                -b toydataset/toy_bins.mat\
                -v toydataset/aleatoric/split/val.txt\
                --training_name aleatoric_NLL

