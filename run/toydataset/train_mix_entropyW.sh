export PYTHONPATH=.
entropy=0.5
# Note: remove random seed in config.py!!
# training set: 1 line
# python src/train_emd_entropy.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train1.txt\
#                 -v toydataset/mix2GNDLarge/split/val1.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name oneline-entropyW$entropy-0\
#                 --entropy $entropy

# python src/train_emd_entropy.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train1.txt\
#                 -v toydataset/mix2GNDLarge/split/val1.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name oneline-entropyW$entropy-1\
#                 --entropy $entropy

# python src/train_emd_entropy.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train1.txt\
#                 -v toydataset/mix2GNDLarge/split/val1.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name oneline-entropyW$entropy-2\
#                 --entropy $entropy

# python src/train_emd_entropy.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train1.txt\
#                 -v toydataset/mix2GNDLarge/split/val1.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name oneline-entropyW$entropy-3\
#                 --entropy $entropy

# python src/train_emd_entropy.py -c toydataset/mix2GNDLarge/twolines.csv\
#                 -i toydataset/mix2GNDLarge/images\
#                 -t toydataset/mix2GNDLarge/split/train1.txt\
#                 -v toydataset/mix2GNDLarge/split/val1.txt\
#                 -b toydataset/toy_bins.mat\
#                 --training_name oneline-entropyW$entropy-4\
#                 --entropy $entropy

# # training set: mix in image
python src/train_emd_entropy.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name mix-entropyW$entropy-0\
                --entropy $entropy\
                --SmoothCoefficient 4 
python src/train_emd_entropy.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name mix-entropyW$entropy-1\
                --entropy $entropy\
                --SmoothCoefficient 4 

python src/train_emd_entropy.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name mix-entropyW$entropy-2\
                --entropy $entropy\
                --SmoothCoefficient 4 

python src/train_emd_entropy.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name mix-entropyW$entropy-3\
                --entropy $entropy\
                --SmoothCoefficient 4 

python src/train_emd_entropy.py -c toydataset/mix2GNDLarge/twolines.csv\
                -i toydataset/mix2GNDLarge/images\
                -t toydataset/mix2GNDLarge/split/train.txt\
                -v toydataset/mix2GNDLarge/split/val1.txt\
                -b toydataset/toy_bins.mat\
                --training_name mix-entropyW$entropy-4\
                --entropy $entropy\
                --SmoothCoefficient 4 
