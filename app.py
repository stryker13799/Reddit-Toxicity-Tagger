from flask import Flask, request
from src.model import ToxicityModel
import src.config as config
from transformers import BertModel,BertConfig,BertTokenizer
import boto3
import os
import torch

############################ Bert Model Path ######################################################
bert_model_path = './bert_model'

############################ Downloading the Model from s3 bucket #################################
filename = "./model1/"
try:
    os.mkdir(filename)
except:
    print("Directory exists")

### Initializing the boto client
ACCESS_KEY = 'AKIAW3CEOBGF6VFBIC7S'
SECRET_KEY = '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'
bucket_name = "toxic-comments19032023"
session = boto3.Session(
    aws_access_key_id=ACCESS_KEY, 
    aws_secret_access_key=SECRET_KEY)

### Downloading the bert model
if not os.path.exists(filename + "best_weights.pt"):
    s3 = session.client("s3")
    s3.download_file(bucket_name,'best_weights', filename + "best_weights.pt")
    print(f"Model downloaded from s3 bucket")
else:
    print(f"Weights Already Exists")
    
############################ Loading the Bert Model and Tokenizer #################################
##Initializing the tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_model_path,do_lower = True)


## Initialzing the the model
bert_config = BertConfig.from_pretrained(bert_model_path)
bert_model = BertModel(config = bert_config)
model = ToxicityModel(bert_model=bert_model)

## Loading the weights
model.load_state_dict(torch.load(filename + "best_weights.pt",map_location = torch.device("cpu")))

###########################  Setting up the FlaskApplication #########################################
app = Flask(__name__)


@app.route("/predict",methods = ['POST'])
def predict_toxicity():
    
    text = request.json['text']
    
    #### Predicitons 
    input = tokenizer(text,padding='max_length',\
                    max_length = config.max_length,return_tensors = "pt")
    out = model(**input)

    #### Returning output
    score = out.detach().squeeze(0).tolist()
    predictions = {lb:sc \
                        for lb,sc in zip(config.labels,score)}
    
    #### Prediction
    return {"prediction":predictions}


if __name__ == "__main__":
    app.run(port=5000)

