export PYTHONPATH=.
# python src/test.py -c toydataset/aleatoric/toydata.csv \
#                 -t toydataset/aleatoric/split/test1.txt \
#                 -i toydataset/aleatoric/images \
#                 -b toydataset/toy_bins.mat \
#                 --checkpoint_path data/toy.txt \
#                 --activation softplus
# python src/test.py -c toydataset/aleatoric/toydata.csv \
#                 -t toydataset/aleatoric/split/test2.txt \
#                 -i toydataset/aleatoric/images \
#                 -b toydataset/toy_bins.mat \
#                 --checkpoint_path data/toy.txt \
#                 --activation softplus
# python src/test.py -c /data/datasets/HLW2/metadata.csv \
#                 -t /data/datasets/HLW2/split/test.txt \
#                 -i /data/datasets/HLW2/images \
#                 -b ./data/bins.mat \
#                 --checkpoint_path data/renorm.txt \
#                 --activation softplus
# python src/test.py -c toydataset/aleatoric/toydata.csv \
#                 -t toydataset/aleatoric/split/test1.txt\
#                 -i toydataset/aleatoric/images \
#                 -b toydataset/toy_bins.mat \
#                 --checkpoint_path data/nll.txt \
#                 --activation softmax

# python src/test.py -c toydataset/aleatoric/toydata.csv \
#                 -t toydataset/aleatoric/split/test2.txt\
#                 -i toydataset/aleatoric/images \
#                 -b toydataset/toy_bins.mat \
#                 --checkpoint_path data/nll.txt \
#                 --activation softmax
python src/test.py -c /data/datasets/HLW2/metadata.csv \
                -t /data/datasets/HLW2/split/test.txt\
                -i /data/datasets/HLW2/images \
                -b data/bins.mat \
                --checkpoint_path data/resnet.txt \
                --activation softmax \
                --log_path logs/test.log