export PYTHONPATH=.
hinge=0
entropy=0.5
# training set: 1 line
# python src/train_EntropyHingeW_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train1.txt\
#                 -v toydataset/mix2GNDLarge/split/val.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name oneline-PW0\
#                 --renormThresh $hinge
# python src/train_EntropyHingeW_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train1.txt\
#                 -v toydataset/mix2GNDLarge/split/val.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name oneline-PW1\
#                 --renormThresh $hinge
# python src/train_EntropyHingeW_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train1.txt\
#                 -v toydataset/mix2GNDLarge/split/val.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name oneline-PW2\
#                 --renormThresh $hinge
# python src/train_EntropyHingeW_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train1.txt\
#                 -v toydataset/mix2GNDLarge/split/val.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name oneline-PW3\
#                 --renormThresh $hinge
# python src/train_EntropyHingeW_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train1.txt\
#                 -v toydataset/mix2GNDLarge/split/val.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name oneline-PW4\
#                 --renormThresh $hinge
# # training set: mix in image
# python src/train_EntropyHingeW_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train.txt\
#                 -v toydataset/mix2GNDLarge/split/val.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name mix-PW0\
#                 --renormThresh $hinge\
#                 --entropy $entropy
# python src/train_EntropyHingeW_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train.txt\
#                 -v toydataset/mix2GNDLarge/split/val.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name mix-PW1\
#                 --renormThresh $hinge\
#                 --entropy $entropy
# python src/train_EntropyHingeW_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train.txt\
#                 -v toydataset/mix2GNDLarge/split/val.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name mix-PW2\
#                 --renormThresh $hinge\
#                 --entropy $entropy
# python src/train_EntropyHingeW_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train.txt\
#                 -v toydataset/mix2GNDLarge/split/val.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name mix-PW3\
#                 --renormThresh $hinge\
#                 --entropy $entropy
# python src/train_EntropyHingeW_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train.txt\
#                 -v toydataset/mix2GNDLarge/split/val.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name mix-PW4\
#                 --renormThresh $hinge\
#                 --entropy $entropy
# training set: mix in GND
python src/train_EntropyHingeW_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train.txt\
                -v toydataset/mix2GNDLarge/split/val.txt\
                -b toydataset/toy_bins.mat\
                --training_name mixGND-PW0\
                --renormThresh $hinge\
                --entropy $entropy\
                --flag
# python src/train_EntropyHingeW_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train.txt\
#                 -v toydataset/mix2GNDLarge/split/val.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name mixGND-PW1\
#                 --renormThresh $hinge\
#                 --entropy $entropy\
#                 --flag
# python src/train_EntropyHingeW_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train.txt\
#                 -v toydataset/mix2GNDLarge/split/val.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name mixGND-PW2\
#                 --renormThresh $hinge\
#                 --entropy $entropy\
#                 --flag
# python src/train_EntropyHingeW_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train.txt\
#                 -v toydataset/mix2GNDLarge/split/val.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name mixGND-PW3\
#                 --renormThresh $hinge\
#                 --entropy $entropy\
#                 --flag
# python src/train_EntropyHingeW_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train.txt\
#                 -v toydataset/mix2GNDLarge/split/val.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name mixGND-PW4\
#                 --renormThresh $hinge\
#                 --entropy $entropy\
#                 --flag