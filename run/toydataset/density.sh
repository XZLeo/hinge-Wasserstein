# run sparisification.sh first
export PYTHONPATH=.
python src/test_resnet_density.py -c /data/datasets/HLW2/metadata.csv \
                -t /data/datasets/HLW2/split/test.txt \
                -i /data/datasets/HLW2/images \
                -b ./data/bins.mat \
                --checkpoint_path data/renorm.txt \
                --activation softplus