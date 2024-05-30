export PYTHONPATH=.
# to test upperbound, use test set as the validation
python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train1.txt\
                -v toydataset/mix2GNDLarge/split/test1.txt\
                -b toydataset/toy_bins.mat\
                --training_name upperbound\
                --renormThresh 0