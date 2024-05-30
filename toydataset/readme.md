The use of toy dataset is described in Sec.4.1 in [https://arxiv.org/pdf/2306.00560.pdf].

## Epistemic uncertainty
- `create_oneline.py` creates training and test set 1; change the $\theta$ distribution to create test set 2.

## Aleatoric uncertainty
- `create_dataset.py` creates `aleatoric`;
- split `aleatoric/test.txt` into one line `aleatoric/test1.txt` and two lines `aleatoric/test2.txt`. 

Note: the distribution of $\theta$ should be the same in three scripts.

For the usage of each script, check the comment.

## File Structure
```
toydataset
├── img             kernel density estmiation
├── readme.md
├── src
│   ├── create_dataset.py
│   ├── create_oneline.py
│   ├── create_twoline.py
│   ├── draw_lines.py
│   ├── get_bin_edges.py
│   └── kde.py
└── aleatoric                 for aleatoric uncertainty
    ├── images
    ├── split
    │   ├── test1.txt   only 1 line/image
    │   ├── test2.txt
    │   ├── test.txt
    │   ├── train.txt   both 1 line/image and 2 lines/image
    │   └── val.txt     same as training set
    └── toydata.csv
```