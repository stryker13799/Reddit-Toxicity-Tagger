from transformers import BertModel
import torch 
import torch.nn as nn


class ToxicityModel(nn.Module):
    def __init__(self, bert_model_path):
        super(ToxicityModel,self).__init__()
        
        self.bert_model = BertModel.from_pretrained(bert_model_path)
        self.l1 = nn.Linear(768,256)  ## Reducing the Vector Dimension
        self.dropout = nn.Dropout(0.2)
        
        ## ['target','severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
        self.toxicity = nn.Linear(256,6)  ## 6 classes
        
    def forward(self,**kwargs):
        
        hc,_ = self.bert_model(**kwargs,return_dict = False)
        x = hc[:,0,:]
        x = self.dropout(self.l1(x))
        x = self.toxicity(x)
        
        return x
        

