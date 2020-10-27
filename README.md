# Query-Session-Matching-CIKM2020

This repo contains the code and data for paper **Query-to-Session Matching: Do NOT Forget History and Future during Response Selection for Multi-Turn Dialogue Systems** in CIKM 2020 by Zhenxin Fu (fuzhenxin95@gmail.com).

## Code
1. Preprocess: ```cd code ; cd utils ; python compose_data.py```
2. The configure is located in main.py (detailed introductions are in main.py)
3. How to train:
```python main.py ubuntu/douban/ecommerce train```
4. How to test:
```python main.py ubuntu/douban/ecommerce test $chenkpoint_file```

## Data Structure
The data sets are located in data directory:
   - the dataset is available at [link](https://drive.google.com/file/d/1nUDrTxWCW6UyZ-bX0iyf9wlAY6H5wq5A/view?usp=sharing). If the link is not available, please send email to me. 
   - train/valid/test.mix : Format of each line in the files: query | history | response | future | label
   - vectors.txt contains the pre-trained word embedding.

## Other
The results of our [KDD](https://dl.acm.org/doi/10.1145/3394486.3403211) paper on this dataset are also shown here
- Ubuntu: MRR 0.7261 R10@1 0.6511 R10@2 0.7944 R10@5 0.9503 R2@1 0.9043
- Douban: MRR 0.7557 R10@1 0.6898 R10@2 0.8233 R10@5 0.9618 R2@1 0.9199
- Ecommerce: MRR 0.8171 R10@1 0.7670 R10@2 0.9005 R10@5 0.9872 R2@1 0.9549

## ACK
The code is developed referring [DAM](https://github.com/baidu/Dialogue/tree/master/DAM).

