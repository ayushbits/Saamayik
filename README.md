# S&#227;mayik: English-Sanskrit Parallel Dataset

## Overview
S&#227;mayik is an English-Sanskrit parallel dataset that captures contemporary usage of Sanskrit, particularly in prose. This dataset comprises of around 53,000 parallel sentence pairs gathered from diverse sources, including spoken content on contemporary world affairs, interpretation of literary works, pedagogical content, and more.

## Data
1. data/final_data/ - contains train, test, dev splits
2. data/\<corpus\> - contains respective corpus such as Mann Ki Baat(mkb), Spoken Tutorials (spoken-tutorials), Gitasopanam (gitasopanam), Bible (bible), NIOS (nios) 

## Evaluation and training scripts
1. Each model folder contains train, evaluation and data generation scripts
2. Fine-tuning and evaluation scripts for IndicTrans2 is directly used from the [original repository](https://github.com/AI4Bharat/IndicTrans2).
## Citation
If you use S&#227;mayik in your research, please cite our paper.


> @misc{maheshwari2023samayik,
      title={S$\={a}$mayik: A Benchmark and Dataset for English-Sanskrit Translation}, 
      author={Ayush Maheshwari and Ashim Gupta and Amrith Krishna and Atul Kumar Singh and Ganesh Ramakrishnan and G. Anil Kumar and Jitin Singla},
      year={2023},
      eprint={2305.14004},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
