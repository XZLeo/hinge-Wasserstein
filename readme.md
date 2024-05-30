<p align="center">
  <h1 align="center"> <ins>Hinge-Wasserstein:</ins> 🎶<br>Estimating Multimodal Aleatoric Uncertainty in Regression Tasks  <br> CVPR 2024 workshop Long Oral</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?hl=en&user=R_6R5mAAAAAJ&view_op=list_works&gmla=ABOlHixSrD6yvittRVaf0Hhl4eTqEUzFOtpndBU6KihFRRWR2c3Nad78FaUfrkDVlx7dMFnxUSdMFKnPfFwhFspz">Ziliang Xiong</a>
    ·
    <a href="https://scholar.google.com/citations?user=dVvOUGYAAAAJ&hl=en&oi=ao">Arvi Jonnarth</a>
    ·
    <a href="https://scholar.google.com/citations?user=9tP8jsUAAAAJ&hl=en&oi=ao"> Abdelrahman Eldesokey</a>
</p>
<p align="center">
    <a href="https://scholar.google.com/citations?user=5sUDSxQAAAAJ&hl=en&oi=ao"> Joakim Johnander</a>
    ·
    <a href="https://scholar.google.com/citations?user=z4aXEBYAAAAJ&hl=en&oi=ao">  Bastian Wandt</a>
    .
    <a href="https://scholar.google.com/citations?user=SZ6jH-4AAAAJ&hl=en&oi=ao">  Per-Erik Forssén</a>
  </p>
  <h2 align="center"><p>
    <a href="https://arxiv.org/pdf/2306.00560" align="center">Paper</a> | 
    <a href="TODO" align="center">Project Page (TODO)</a>
  </p></h2>
  <div align="center"></div>
</p>
<p align="center">
    <img src="assets/matches.jpg" alt="example" width=80%>
    <br>
    <em>The DeDoDe detector learns to detect 3D consistent repeatable keypoints, which the DeDoDe descriptor learns to match. The result is a powerful decoupled local feature matcher.</em>
</p>

## Dataset
### Sythetic Dataset
### Horizon Line in the Wild

## Training
1. Train with hinge-Wasserstein
```
python src/train_HingeW.py -c path2hlw/metadata.csv -i path2hlw/images -t path2hlw/split/train.txt
-v path2hlw/split/val.txt
--training_name name
--renormThresh 0.01
```
2. Train with NLL
```
python src/train_nll.py
```
3. Train with Synthetic Dataset with bimodal ground truth
```
python src/train_HingeW_twoline.py
```
## Testing
1. Test with metrics including, AUC-HE, AUSE, NLL, RMSE, ECE
```
python src/test.py
```
2. Plot density
```
python src/test_density.py
```

## Inference

