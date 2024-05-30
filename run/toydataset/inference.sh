export PYTHONPATH=.
python src/inference.py -c /data/datasets/HLW2/metadata.csv\
                        --bin_edges_path data/bins.mat\
                        --model_weights_path pretrained_model/renorm.02/best.pth.tar \
                        --image_path data/inference.txt
