export PYTHONPATH=.
# Note: remove random seed in config.py!!
# training set: 1 line
# python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train1.txt\
#                 -v toydataset/mix2GNDLarge/split/val1.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name oneline-softHW-pow4-0
# python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train1.txt\
#                 -v toydataset/mix2GNDLarge/split/val1.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name oneline-softHW-pow4-1
python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train1.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name oneline-softHW-pow4-2
python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train1.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name oneline-softHW-pow4-3
python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train1.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name oneline-softHW-pow4-4
# training set: mix in image
python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name mix-softHW-pow4-0
python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name mix-softHW-pow4-1
python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name mix-softHW-pow4-2
python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name mix-softHW-pow4-3
python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name mix-softHW-pow4-4
# training set: mix in GND
# python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train.txt\
#                 -v toydataset/mix2GNDLarge/split/val1.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name mixGND-softHW-pow4-0\
#                 --flag
# python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train.txt\
#                 -v toydataset/mix2GNDLarge/split/val1.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name mixGND-softHW-pow4-1\
#                 --flag
# python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train.txt\
#                 -v toydataset/mix2GNDLarge/split/val1.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name mixGND-softHW-pow4-2\
#                 --flag
# python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train.txt\
#                 -v toydataset/mix2GNDLarge/split/val1.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name mixGND-softHW-pow4-3\
#                 --flag
# python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                # -i toydataset/mix2GNDLarge/images\
                # -t toydataset/mix2GNDLarge/split/train.txt\
                # -v toydataset/mix2GNDLarge/split/val1.txt\
                # -b toydataset/toy_bins.mat\
                # --training_name mixGND-softHW-pow4-4\
                # --flag

