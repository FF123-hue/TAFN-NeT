
### 1. Prepare the datasets. 
(1) SYSU-MM01 Dataset

   - run `python pre_process_sysu.py` to pepare the dataset, the training data will be stored in ".npy" format.
 
(2) LLCM Dataset

### 2. Training.
Train a model by:
```
python train.py --dataset llcm --gpu 0
python train.py --dataset sysu --gpu 0

```
--dataset: which dataset "llcm" or "sysu".

--gpu: which gpu to run.

You may need mannully define the data path first.

Parameters: More parameters can be found in the script.

### 3. Testing.
Test a model on LLCM, SYSU-MM01 or RegDB dataset by
```
python test.py --mode all --tvsearch True --resume 'model_path' --gpu 1 --dataset llcm

```
--dataset: which dataset "llcm" or "sysu".

--mode: "all" or "indoor" all search or indoor search.

--tvsearch: whether thermal to visible search.

--resume: the saved model path.

--gpu: which gpu to run.


