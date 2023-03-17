from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd


class ToxicityDataset(Dataset):
    def __init__(self,data_path,tokenizer,max_length = 128):
        ## Initializing some variables in the constructor
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = 128
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        
        ## Accessing the single item
        item = self.data.iloc[idx]
        
        ## The input comment text
        comment_text = item['comment_text']
        
        ## The output labels
        toxicity = item['target_label']
        severe_toxicity = item['severe_toxicity']
        obscene = item['obscene']
        identity_attack = item['identity_attack']
        insult = item['insult']
        threat = item['threat']
        
        ## tokenizing the text
        input_tensors = self.tokenizer(comment_text,padding="max_length",\
                                    max_length=self.max_length,truncation=True,\
                                         return_tensors = "pt")
        
        ## Reducing a dimension for each key
        input_tensors = {k:v.squeeze(0) for k,v in input_tensors.items()}
        
        ## Processing the output labels
        labels = [toxicity,severe_toxicity,obscene,identity_attack,insult,threat]
        labels = torch.tensor(labels,dtype=torch.float32)
        
        ## returning the result
        return {"input":input_tensors,"labels":labels}