export PYTHONPATH=.
python src/train_HingeEmd.py -c toydataset/clean/twolines.csv\
                -i toydataset/clean/images\
                -t toydataset/clean/split/train.txt\
                -b toydataset/toy_bins.mat\
                -v toydataset/clean/split/val.txt\
                --training_name clean_PW\
                --renormThresh 0

