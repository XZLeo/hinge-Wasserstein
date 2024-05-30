export PYTHONPATH=.
# training set: 1 line
# python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train.txt\
#                 -v toydataset/mix2GNDLarge/split/val1.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name mix-HW.015-0\
#                 --renormThresh 0.015
python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name test\
                --renormThresh 0.015\
                --flag
# 唯一的区别是换了train.txt