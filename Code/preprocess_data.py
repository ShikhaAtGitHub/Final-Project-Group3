from datasets import load_dataset
from pprint import pprint
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle


# Download the SQuAD Dataset
train_dataset = load_dataset('squad', split = 'train')
valid_dataset = load_dataset('squad', split = 'validation')

# Print sample from validation dataset
# sample_validation_dataset = next(iter(valid_dataset))
# pprint(sample_validation_dataset)

# Extract context, question and answer from the sample
# context = sample_validation_dataset['context']
# question = sample_validation_dataset['question']
# answer = sample_validation_dataset['answers']['text'][0]

# print("context: ", context)
# print("question: ", question)
# print("answer: ", answer)

# Set pandas display option to view entire context
# pd.set_option("display.max_colwidth", -1)

# Initialize empty dataframes for training and validation sets
df_train = pd.DataFrame(columns= ['context', 'answer', 'question'])
df_valid = pd.DataFrame(columns= ['context', 'answer', 'question'])

# print(df_valid)
# print(df_train)

# Counters for the number of long and short answers
count_long = 0
count_short = 0

# Process the training dataset
for index,val in enumerate(tqdm(train_dataset)):
    context = val['context']
    question = val['question']
    answer = val['answers']['text'][0]
    no_of_words = len(answer.split())
    # Skip appending the dataframe based on answer length
    if no_of_words >= 7:
        count_long = count_long + 1
        continue
    # Populate the training set
    else:
        df_train.loc[count_short] = [context] + [answer] + [question]
        count_short = count_short + 1

# print("count_long train dataset:", count_long)
# print("count_short train dataset:", count_short)

# Reset counters for the validation dataset
count_long = 0
count_short = 0

for index,val in enumerate(tqdm(valid_dataset)):
    context = val['context']
    question = val['question']
    answer = val['answers']['text'][0]
    no_of_words = len(answer.split())
    # Skip appending the dataframe based on answer length
    if no_of_words >= 7:
        count_long = count_long + 1
        continue
    # Populate the validation set
    else:
        df_valid.loc[count_short] = [context] + [answer] + [question]
        count_short = count_short + 1

# print("count_long validation dataset:", count_long)
# print("count_short validation dataset:", count_short)

# Shuffle the dataframes
df_train = shuffle(df_train)
df_valid = shuffle(df_valid)

# print(df_train.shape)
# print(df_valid.shape)

# print(df_train.head())
# print(df_valid.head())

# Define file paths for saving the CSV files
train_save_path = "./dataset/squad_t5_train.csv"
valid_save_path = "./dataset/squad_t5_valid.csv"

# Save the dataframes to CSV
df_train.to_csv(train_save_path, index = False)
df_valid.to_csv(valid_save_path, index = False)