export PYTHONPATH=.
python src/inference2line.py -c toydataset/mix2GND/twolines.csv \
                -i data/test.txt \
                -b /home/zilxi06/debug/toydataset/toy_bins.mat \
                --checkpoint_path /work2/hlw/samples/oneline-HP.015-0/epoch_150.pth.tar
