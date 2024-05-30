export PYTHONPATH=.
python src/train_HingePred_twoline.py -c toydataset/mix2GND/twolines.csv\
                -i toydataset/mix2GND/images\
                -t toydataset/mix2GND/split/train.txt\
                -b toydataset/toy_bins.mat\
                --training_name oneline-HP.015-0\
                --renormThresh 0.015
python src/train_HingePred_twoline.py -c toydataset/mix2GND/twolines.csv\
                -i toydataset/mix2GND/images\
                -t toydataset/mix2GND/split/train.txt\
                -b toydataset/toy_bins.mat\
                --training_name oneline-HP.015-1\
                --renormThresh 0.015
python src/train_HingePred_twoline.py -c toydataset/mix2GND/twolines.csv\
                -i toydataset/mix2GND/images\
                -t toydataset/mix2GND/split/train.txt\
                -b toydataset/toy_bins.mat\
                --training_name oneline-HP.015-2\
                --renormThresh 0.015
python src/train_HingePred_twoline.py -c toydataset/mix2GND/twolines.csv\
                -i toydataset/mix2GND/images\
                -t toydataset/mix2GND/split/train.txt\
                -b toydataset/toy_bins.mat\
                --training_name oneline-HP.015-3\
                --renormThresh 0.015
python src/train_HingePred_twoline.py -c toydataset/mix2GND/twolines.csv\
                -i toydataset/mix2GND/images\
                -t toydataset/mix2GND/split/train.txt\
                -b toydataset/toy_bins.mat\
                --training_name oneline-HP.015-4\
                --renormThresh 0.015

