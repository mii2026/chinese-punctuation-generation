# chinese-punctuation-generation

## Environment
We run this code in this environment.
- python=3.8.0
- CUDA 11.6
- torch 1.13.1
- transformers 4.26.1
- tokenizers 0.13.2
- numpy 1.23.5
- pandas 1.3.5
- scikit-learn 0.23.2
   
You can set the environment by installing anaconda and running the command lines.
```
conda create -n chinese python=3.8.0
conda activate chinese
pip install -r requirements.txt
python -m ipykernel install --user --name chinese --display-name chinese
```

## Datasets
Our preprocessed data is saved in ```data/``` folder. List of data is as followed.
- Preprocessed from dataset we make
  - our_test.txt
  - our_train1.txt
  - our_train2.txt
  - our_train3.txt
  - our_valid.txt
- Preprocessed from iwslt2012 dataset
  - test_iwslt.txt
  - train_iwslt.txt
  - valid_iwslt.txt

## Preprocessing new data
To preprocess new data, you can run ```preprocess_data_our.ipynb``` file. Details are in the file.

## Training

To train the model, run the command line.
```
python train.py
```
To change model configuration, you can change the ```config.json``` file. Details of configuration is as bellow.
- sequence_len: Length of input
- loss function: Loss function using until training. (Three options:"crossentropy", "focal", "scl")
- model_save_path: Path to save model
- train_data_path: Path of train datasets. You can write more than one.
- valid_data_path": Path of valid datasets. You can write more than one.

## Evaluation
To evaluate the model, run ```evaluate.ipynb```. Details are in the file.

## References
- This code follow the model architechture of [Word-level BERT-CNN-RNN Model for Chinese Punctuation Restoration](https://ieeexplore.ieee.org/document/9344889).
- ```focal_loss.py``` and ```scl_loss.py``` is modified from https://github.com/hqsiswiliam/punctuation-restoration-scl.