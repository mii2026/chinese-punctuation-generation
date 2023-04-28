from torch.utils.data import TensorDataset
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast

def preprocessing(datasets, sequence_len):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese', do_lower_case=True)
    punc_enc = {'O': 0,',': 1,'。': 2,'!': 3,'?': 4,';': 5,':': 6,'“' : 7,'”' : 8,'…' : 9,'—' : 10,'、' : 11,'·' : 12,'《' : 13,'》' : 14}
    words = []
    targets = []
    for dataset in datasets:
        for data in tqdm(dataset):
            try:
                x, target = data.split()
                punc = target.strip()
                x = tokenizer.tokenize(x)
                x = tokenizer.convert_tokens_to_ids(x)
                target = [punc_enc[punc]]
                if len(x)<1:
                    continue
                if len(x)>1:
                    target = [0]*(len(x)-1) + target
                words+=x
                targets+=target
            except:
                print(data)

    words = torch.tensor(words).long()
    targets = torch.tensor(targets).long()
    assert len(words) == len(targets)

    remain = len(words) % sequence_len
    words = words[:-remain].view(-1, sequence_len)
    targets = targets[:-remain].view(-1, sequence_len)

    dataset = TensorDataset(words, targets)
    return dataset

if __name__ == '__main__':
    datasets = []
    with open('data/valid_iwslt.txt', 'r', encoding='utf-8') as f:
        data = f.readlines()
    datasets.append(data)
    with open('data/our_valid.txt', 'r', encoding='utf-8') as f:
        data = f.readlines()
    datasets.append(data)

    dataset = preprocessing(datasets, 100)
    print(dataset[0])