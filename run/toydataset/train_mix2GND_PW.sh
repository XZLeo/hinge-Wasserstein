export PYTHONPATH=.
# training set: 1 line
# # python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
# #                 -i toydataset/mix2GNDLarge/images\
# #                 -t toydataset/mix2GNDLarge/split/train1.txt\
# #                 -v toydataset/mix2GNDLarge/split/val1.txt\
# #                 -b toydataset/toy_bins.mat\
# #                 --training_name oneline-PW0\
# #                 --renormThresh 0
# # python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
# #                 -i toydataset/mix2GNDLarge/images\
# #                 -t toydataset/mix2GNDLarge/split/train1.txt\
# #                 -v toydataset/mix2GNDLarge/split/val1.txt\
# #                 -b toydataset/toy_bins.mat\
# #                 --training_name oneline-PW1\
# #                 --renormThresh 0
# # python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
# #                 -i toydataset/mix2GNDLarge/images\
# #                 -t toydataset/mix2GNDLarge/split/train1.txt\
# #                 -v toydataset/mix2GNDLarge/split/val1.txt\
# #                 -b toydataset/toy_bins.mat\
# #                 --training_name oneline-PW2\
# #                 --renormThresh 0
# # python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
# #                 -i toydataset/mix2GNDLarge/images\
# #                 -t toydataset/mix2GNDLarge/split/train1.txt\
# #                 -v toydataset/mix2GNDLarge/split/val1.txt\
# #                 -b toydataset/toy_bins.mat\
# #                 --training_name oneline-PW3\
# #                 --renormThresh 0
# # python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
# #                 -i toydataset/mix2GNDLarge/images\
# #                 -t toydataset/mix2GNDLarge/split/train1.txt\
# #                 -v toydataset/mix2GNDLarge/split/val1.txt\
# #                 -b toydataset/toy_bins.mat\
# #                 --training_name oneline-PW4\
# #                 --renormThresh 0
# # # training set: mix in image
# python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train.txt\
#                 -v toydataset/mix2GNDLarge/split/val1.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name mix-PW0\
#                 --renormThresh 0
# python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train.txt\
#                 -v toydataset/mix2GNDLarge/split/val1.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name mix-PW1\
#                 --renormThresh 0
python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name mix-PW2\
                --renormThresh 0
python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name mix-PW3\
                --renormThresh 0
python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name mix-PW4\
                --renormThresh 0
# training set: mix in GND
python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name mixGND-PW0\
                --renormThresh 0\
                --flag
python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name mixGND-PW1\
                --renormThresh 0\
                --flag
python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name mixGND-PW2\
                --renormThresh 0\
                --flag
python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name mixGND-PW3\
                --renormThresh 0\
                --flag
python src/train_HingeEmd_twoline.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name mixGND-PW4\
                --renormThresh 0\
                --flag