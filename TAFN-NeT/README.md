
### 1. Prepare the dataset. 
   SYSU-MM01 Dataset

   - Can be downloaded from https://pan.quark.cn/s/6a7661005a50  Extraction code: iHcc

   - run `python pre_process_sysu.py` to pepare the dataset, the training data will be stored in ".npy" format.
 
### 2. Training.
Train a model by:
```
python train.py --dataset sysu --gpu 0

```

--gpu: which gpu to run.

You may need mannully define the data path first.

Parameters: More parameters can be found in the script.

### 3. Testing.
Test a model on SYSU-MM01 dataset by
```
python test.py --mode all --tvsearch True --resume 'model_path' --gpu 1 --dataset sysu

```

--mode: "all" or "indoor" all search or indoor search.

--resume: the saved model path.

--gpu: which gpu to run.

