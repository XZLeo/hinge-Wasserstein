export PYTHONPATH=.
# Note: remove random seed in config.py!!
# training set: 1 line
python src/train_SoftHingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train1.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name oneline-softHW-0
python src/train_SoftHingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train1.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name oneline-softHW-1
python src/train_SoftHingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train1.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name oneline-softHW-2
python src/train_SoftHingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train1.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name oneline-softHW-3
python src/train_SoftHingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train1.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name oneline-softHW-4
# training set: mix in image
python src/train_SoftHingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name mix-softHW-0
python src/train_SoftHingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name mix-softHW-1
python src/train_SoftHingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name mix-softHW-2
python src/train_SoftHingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name mix-softHW-3
python src/train_SoftHingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name mix-softHW-4
# training set: mix in GND
# python src/train_SoftHingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train.txt\
#                 -v toydataset/mix2GNDLarge/split/val1.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name mixGND-softHW-0\
#                 --flag
# python src/train_SoftHingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train.txt\
#                 -v toydataset/mix2GNDLarge/split/val1.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name mixGND-softHW-1\
#                 --flag
# python src/train_SoftHingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train.txt\
#                 -v toydataset/mix2GNDLarge/split/val1.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name mixGND-softHW-2\
#                 --flag
# python src/train_SoftHingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train.txt\
#                 -v toydataset/mix2GNDLarge/split/val1.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name mixGND-softHW-3\
#                 --flag
# python src/train_SoftHingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                # -i toydataset/mix2GNDLarge/images\
                # -t toydataset/mix2GNDLarge/split/train.txt\
                # -v toydataset/mix2GNDLarge/split/val1.txt\
                # -b toydataset/toy_bins.mat\
                # --training_name mixGND-softHW-4\
                # --flag

