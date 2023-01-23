# AGPF-FSFG

## Official codes for [Learning Attention-Guided Pyramidal Features for Few-shot Fine-grained Recognition](https://www.sciencedirect.com/science/article/pii/S0031320322002734) (Pattern Recognition, 2022) [![IEEE Page](https://img.shields.io/badge/Elsevier-PR.2022.108792-green?style=flat-square)](https://www.sciencedirect.com/science/article/pii/S0031320322002734)
### By [Hao Tang](https://cser-tang-hao.github.io/), Chengcheng Yuan, Zechao Li, and Jinhui Tang
### Extension of Conference Paper (IJCAI 2021 LTDL Workshop Best Paper Award)
## Enviroment
 - `Python3`
 - [`Pytorch`](http://pytorch.org/) >= 1.6.0 
 - `CUDA` = 10.2
 - `json`


## Datasets
### CUB
* Change directory to `./filelists/CUB`
* run `source ./download_CUB.sh`

### FGVC
* Change directory to `./filelists/Aircrafts`
* change variable data_dir in Aircrafts_prepare_csv.py
* run `python Aircrafts_prepare_csv.py`

### StanfordCars
* Change directory to `./filelists/StanfordCars`
* change variable data_dir in StanforCar_prepare_csv.py
* run `python StanforCar_prepare_csv.py`

### StanfordDogs
* Change directory to `./filelists/StanfordDogs`
* change variable data_dir in StanfordDog_prepare_csv.py
* run `python StanfordDog_prepare_csv.py`

### Self-defined setting
* Require three data split json file: `base.json`, `val.json`, `novel.json` for each dataset  
* The format should follow   
{"label_names": `["class0","class1",...]`, "image_names": `["filepath1","filepath2",...]`,"image_labels": `[l1,l2,l3,...]`}  
See test.json for reference
* Put these file in the same folder and change data_dir `['DATASETNAME']` in configs.py to the folder path  

## Train
Run  
```python ./train.py --train_n_way [TRAIN_N_WAY] --test_n_way [TEST_N_WAY] --n_shot [K_SHOT]  --stop_epoch [EPOCHS]  --dataset [DATASETNAME] --model [BACKBONENAME] --method [METHODNAME] --num_classes [NUM_CLASSES] --train_aug --apcnn [--OPTIONARG]```

For example,  
```python ./train.py --train_n_way 5 --test_n_way 5 --n_shot 1  --stop_epoch 120  --dataset CUB --model Conv4 --method protonet --num_classes 200 --train_aug --apcnn``` 

[comment]: <> (1 epoch = 500 eposides. )
Commands below follow this example, and please refer to `io_utils.py` for additional options.

## Save features
Save the extracted feature before the classifaction layer to increase test speed.  
```python ./save_features.py --train_n_way 5 --test_n_way 5 --n_shot 1  --dataset CUB --model Conv4 --method protonet --train_aug --apcnn --num_classes 200  ```

## Test

```python ./test.py --train_n_way 5 --test_n_way 5 --n_shot 1  --dataset CUB --model Conv4 --method protonet --train_aug --apcnn --num_classes 200 ```

## Results
* The test results will be recorded in `./record/results.txt`

## Other related papers
  ```bibtex
     @inproceedings{TianTD21,
      author    = {Sheng Tian and Hao Tang and Longquan Dai},
      title     = {Coupled Patch Similarity Network FOR One-Shot Fine-Grained Image Recognition},
      booktitle = {ICIP},
      pages     = {2478--2482},
      year      = {2021}，
      publisher = {{IEEE}}
    }
    
  ```
  ```bibtex
    @article{zha2023boosting,
       title={Boosting Few-shot Fine-grained Recognition with Background Suppression and Foreground Alignment},
       author={Zha, Zican and Tang, Hao and Sun, Yunlian and Tang, Jinhui},
       journal={IEEE Transactions on Circuits and Systems for Video Technology},
       year={2023},
       publisher={IEEE},
       doi={10.1109/TCSVT.2023.3236636}
    }
  ```

## Citation
If this work is useful in your research, please cite 

```bibtex
  @article{TangYLT22,
          author    = {Hao Tang and Chengcheng Yuan and Zechao Li and Jinhui Tang},
          title     = {Learning attention-guided pyramidal features for few-shot fine-grained recognition},
          journal   = {Pattern Recognit.},
          volume    = {130},
          pages     = {108792},
          year      = {2022}
        }
  ```

## References
This implementation builds upon several open-source codes. Specifically, we have modified and integrated the following codes into this repository:

*  [CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot) 
*  [AP-CNN](https://github.com/PRIS-CV/AP-CNN_Pytorch-master) 

