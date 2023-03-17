import mlflow
from transformers import BertTokenizer,BertModel,BertConfig,get_linear_schedule_with_warmup
import config
from model import ToxicityModel
from dataset import ToxicityDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm

def train():
    
    ########### Setting the device #################
    device = "cuda" if torch.cuda.is_available() else "cpu"


    ########### Setting up the tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_path)
    
    
    ########### Loading dataloaders
    ### train
    train_dataset = ToxicityDataset(config.train_data_path,tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True)

    ### Validation
    valid_dataset = ToxicityDataset(config.validation_data_path,tokenizer=tokenizer)
    valid_dataloader = DataLoader(valid_dataset,batch_size=config.batch_size)
    
    
    ### Setting up the model
    bert_model = BertModel.from_pretrained(config.bert_model_path)
    model = ToxicityModel(bert_model=bert_model)
    model.to(device)
    print(f"Model moved to {device}")
    
    ########### Setting up the optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(),lr = config.lr)
    
    total_steps = (len(train_dataloader) //config.batch_size) * config.epochs
    num_warmup_steps = total_steps//5
    scheduler = get_linear_schedule_with_warmup(
                                                optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)
    
    
    ########### Setting up the loss function
    loss_fn = nn.BCELoss()
    
    
    ########### logging parameters on mlflow
    mlflow.log_params({
        "epochs":config.epochs,
        "batch_size":config.batch_size,
        "lr":config.lr,
        "warmup_steps":num_warmup_steps,
        "max_length":config.max_length
    })
    
    
    ########### Training loop ##############
    
    log_train_loss = []
    log_val_loss = []
    best_loss = -999

    for epoch in range(config.epochs):
        
        train_losses = 0
        valid_losses = 0 
        model.train()
        for batch in tqdm(train_dataloader):
            
            batch['input'] = {k:v.to(device) for k,v in batch['input'].items()}
            batch['labels'] = batch['labels'].to(device)
            
            loss = model.training_step(batch['input'],batch['labels'],loss_fn)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_losses+=loss.detach().cpu().item()
            
            
        log_train_loss.append(train_losses/len(train_dataloader))

        model.eval()
        for batch in tqdm(valid_dataloader):
            
            batch['input'] = {k:v.to(device) for k,v in batch['input'].items()}
            batch['labels'] = batch['labels'].to(device)        
            loss = model.training_step(batch['input'],batch['labels'],loss_fn)

            valid_losses+=loss.detach().cpu().item()
            
        log_val_loss.append(valid_losses/len(valid_dataloader))


        if log_val_loss[-1] < best_loss:
            best_loss = log_val_loss[-1]
            torch.save(model.parameters(),"../model/best.pt")

        if (epoch % 5 == 0):
            print(f"Epochs :{epoch}  ->  Train loss : {log_train_loss[-1]}  Valid loss : {log_val_loss[-1]}")
           
        ### logging mlflow
        mlflow.log_metrics({
                    "train_loss":log_train_loss[-1],"valid_loss":log_val_loss[-1]
                            },step = epoch)
    
    history = {"train":log_train_loss,"valid":log_val_loss}
    
    return history


if __name__ == "__main__":
    
    mlflow.set_experiment("testing1")
    with mlflow.start_run():
        train()
        mlflow.log_artifacts("../model","best_weights")
            
    
    
    