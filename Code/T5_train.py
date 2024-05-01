import argparse
import pandas as pd
import copy
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer
)

train_file_path = 'squad_t5_train.csv'
validation_file_path = 'squad_t5_val.csv'

t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

# preparing dataset

class QuestionGenerationDataset(Dataset):
    def __init__(self, tokenizer, filepath, max_len_inp = 512, max_len_out = 96):
        self.path = filepath
        self.passage_column = "context"
        self.answer = "answer"
        self.question = "question"

        self.data = pd.read_csv(self.path, nrows=50000)
        self.max_len_input = max_len_inp
        self.max_len_output = max_len_out
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.skippedcount = 0
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        print(1111111)
        print(self.targets[index])
        target_ids = self.targets[index]["input_ids"].squeeze()
        print(222222222)
        print(target_ids)

        src_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        labels = copy.deepcopy(target_ids)
        labels[labels==0] = -100

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "labels": labels}

    def _build(self):
        for idx in tqdm(range(len(self.data))):
            passage, answer, target = self.data.loc[idx, self.passage_column], self.data.loc[idx, self.answer], \
            self.data.loc[idx, self.question]

            input_ = "context: %s  answer: %s </s>" % (passage, answer)
            target = "question: %s </s>" % (str(target))

            # get encoding length of input, if it is greater than self.max_len skip it
            test_input_encoding = self.tokenizer.encode_plus(input_,
                                                             truncation=False,
                                                             return_tensors="pt")

            length_of_input_encoding = len(test_input_encoding['input_ids'][0])

            if length_of_input_encoding > self.max_len_input:
                self.skippedcount = self.skippedcount + 1
                continue

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], max_length=self.max_len_input, pad_to_max_length=True, return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len_output, pad_to_max_length=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)

# train_dataset = QuestionGenerationDataset(t5_tokenizer,train_file_path)
# validation_dataset = QuestionGenerationDataset(t5_tokenizer,validation_file_path)

class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams, t5model, t5tokenizer):
        super(T5FineTuner, self).__init__()
        self.save_hyperparameters(hparams)
        # self.hparams = hparams
        self.model = t5model
        self.tokenizer = t5tokenizer

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None,
                lm_labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids=batch["target_ids"],
            decoder_attention_mask=batch["target_mask"],
            lm_labels=batch["labels"]
        )

        loss = outputs[0]
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids=batch["target_ids"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )

        loss = outputs[0]
        self.log("val_loss", loss)
        return loss

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(validation_dataset, batch_size=self.hparams.batch_size, num_workers=4)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=3e-4, eps=1e-8)
        return optimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and save model and tokenizer paths')
    parser.add_argument('--model_path', type=str, help='Path to save the model')
    parser.add_argument('--tokenizer_path', type=str, help='Path to save the tokenizer')
    print('Generating Dataset!')
    pl.seed_everything(42)
    train_dataset = QuestionGenerationDataset(t5_tokenizer, train_file_path)
    validation_dataset = QuestionGenerationDataset(t5_tokenizer, validation_file_path)
    print('Dataset Generation Done!')
    print('Start Training the model!')
    # Now train the model
    args_dict = dict(
        batch_size=4,
    )
    args = argparse.Namespace(**args_dict)
    model = T5FineTuner(args, t5_model, t5_tokenizer)
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model)
    print("Training completed!!!")

    print("Saving model")
    # save_path_model = '/Users/shikharaikhare/Documents/Course_work/NLP/NLP_Final_Project/Final-Project-Group3/Code/model'
    # save_path_tokenizer = '/Users/shikharaikhare/Documents/Course_work/NLP/NLP_Final_Project/Final-Project-Group3/Code/tokenizer'
    model.model.save_pretrained(args.model_path)
    t5_tokenizer.save_pretrained(args.tokenizer_path)
    print('Model saved')