# MPCCI
Mixed Prototype Correction for Causal Inference in Medical Image Classification

Dataset Preparation
---
CT COVID-19: link：https://www.kaggle.com/datasets/maedemaftouni/large-covid19-ct-slice-dataset. <br>
BUSI: link: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset. <br>
Please prepare the data as described in the following link: https://dl.acm.org/doi/abs/10.1145/3664647.3681395. <br>

How to run
---
python main.py --device cuda:0 --dataset ct

Citation
---
@inproceedings{10.1145/3664647.3681395,
author = {Zhang, Yajie and Huang, Zhi-An and Hong, Zhiliang and Wu, Songsong and Wu, Jibin and Tan, Kay Chen},
title = {Mixed Prototype Correction for Causal Inference in Medical Image Classification},
year = {2024},
isbn = {9798400706868},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3664647.3681395},
doi = {10.1145/3664647.3681395},
booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
pages = {4377–4386},
numpages = {10},
keywords = {causal inference, disease diagnosis, front-door adjustment, multi-view prototype learning},
location = {Melbourne VIC, Australia},
series = {MM '24}
}

Contact
---
If you have any questions, please contact rubyzhangyajie@gmail.com 
