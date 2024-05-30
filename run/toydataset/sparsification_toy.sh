export PYTHONPATH=.
python src/test.py -c toydataset/mix2GNDLarge/twolines.csv \
                -t toydataset/mix2GNDLarge/split/test2.txt \
                -i toydataset/mix2GNDLarge/images \
                -b toydataset/toy_bins.mat \
                --checkpoint_path data/toy-PW.txt \
                --activation softplus
