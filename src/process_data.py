import pandas as pd
import numpy as np
import config
import os
import clean_text
import argparse
import pickle
from keras.preprocessing.text import Tokenizer

def process_data(
                 data_path,
                 save_dir):
    
    """ The function is to process and clean the raw text and split into train, test, validate

    Args:
        data_path (str): The path to comments dataset
        save_dir (str): The path to store preprocessed datasets along the with spliting
    """
    
    data = pd.read_csv(data_path,index_col=0)
    
    ## Extracting the relevant columns
    relevant_df = data[config.columnns]
    
    
    ## Binariziation of target column
    relevant_df['target'] =np.where(relevant_df['target'] >=0.5,1,0) 
    
    
    ## Clean the comments
    relevant_df['comment_text'] = relevant_df['comment_text'].apply(lambda x: clean_text.preprocess_text(x))
    print("All the text is cleaned")  ## As a logging step. In future, we can add logger here
    
    
    ## Tokenize the text
    tokenizer = Tokenizer(num_words=config.max_features, filters='',lower=False)
    tokenizer.fit_on_texts((list(relevant_df['comment_text'])))
    
    
    ## Splitting the text into train test validate split 
    train, validate, test = np.split(relevant_df.sample(frac=1), \
                                [int(config.train_size * len(relevant_df)), \
                                int((config.train_size+config.test_size) * len(relevant_df))])
    
    
    ## Storing the data files into the local system
    df_train = pd.DataFrame(train)
    df_validate = pd.DataFrame(validate)
    df_test = pd.DataFrame(test)
    
    df_train.to_csv(os.path.join(save_dir,"train_split.csv"))
    df_validate.to_csv(os.path.join(save_dir,"validate.csv"))
    df_test.to_csv(os.path.join(save_dir,"test.csv"))
    
    
    ## Storing the tokenizer on the local file system
    with open(config.tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    
    ## Path of data file
    parser.add_argument('--data_dir', type=str, required=True)

    ## Path for the save directory
    parser.add_argument('--save_dir', type=str, required=True)
    
    
    args = parser.parse_args()
    
    process_data(args.data_dir,args.save_dir)
    print(f"Data Saved to {args.save_dir}")
    
    