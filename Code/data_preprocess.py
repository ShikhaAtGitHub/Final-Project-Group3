from datasets import load_dataset
from pprint import pprint
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.utils import shuffle

train_dataset = load_dataset('squad', split='train')
valid_dataset = load_dataset('squad', split='validation')

print(train_dataset)

# checking the structure if the dataset
sample_validation_dataset = next(iter(valid_dataset))
pprint (sample_validation_dataset)

# extracting the context of the dataset
context = sample_validation_dataset['context']
question = sample_validation_dataset['question']
answer = sample_validation_dataset['answers']['text'][0]

print ("context: ",context)
print ("question: ",question)
print ("answer: ",answer)

# create a dataframe for train and validation
df_train = pd.DataFrame(columns = ['context', 'answer','question'])
df_validation = pd.DataFrame(columns = ['context', 'answer','question'])
print (df_validation)
print (df_train)

def create_df(dataset, df):
    count_long = 0
    count_short = 0

    for index,val in enumerate(dataset):
        passage = val['context']
        question = val['question']
        answer = val['answers']['text'][0]
        no_of_words = len(answer.split())
        if no_of_words >= 7:
            count_long = count_long + 1
            continue
        else:
            df.loc[count_short]= [passage] + [answer] + [question]
            count_short = count_short + 1

    print ("count_long train dataset: ",count_long)
    print ("count_short train dataset: ",count_short)
    return df

df_train = create_df(train_dataset, df_train)
df_validation = create_df(valid_dataset, df_validation)

print(df_train.head())
print(df_validation.head())

df_train = shuffle(df_train)
df_validation = shuffle(df_validation)

print(df_train.shape)
print(df_validation.shape)

# save both train and validation in Data folder
train_save_path = 'squad_t5_train.csv'
validation_save_path = 'squad_t5_val.csv'
df_train.to_csv(train_save_path, index = False)
df_validation.to_csv(validation_save_path, index = False)


