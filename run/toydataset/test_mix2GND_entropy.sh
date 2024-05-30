export PYTHONPATH=.
# mix-entropyW
python src/test.py -c toydataset/mix2GNDLarge/twolines.csv \
                -t toydataset/mix2GNDLarge/split/test1.txt \
                -i toydataset/mix2GNDLarge/images \
                -b toydataset/toy_bins.mat \
                --checkpoint_path data/toy-mix-entropyW.txt \
                --activation softplus
python src/test.py -c toydataset/mix2GNDLarge/twolines.csv \
                -t toydataset/mix2GNDLarge/split/test2.txt \
                -i toydataset/mix2GNDLarge/images \
                -b toydataset/toy_bins.mat \
                --checkpoint_path data/toy-mix-entropyW.txt \
                --activation softplus
python src/test.py -c toydataset/mix2GNDLarge/twolines.csv \
                -t toydataset/mix2GNDLarge/split/test.txt \
                -i toydataset/mix2GNDLarge/images \
                -b toydataset/toy_bins.mat \
                --checkpoint_path data/toy-mix-entropyW.txt \
                --activation softplus
