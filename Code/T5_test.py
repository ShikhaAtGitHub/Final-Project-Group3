import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import argparse

# trained_model_path = '/Users/shikharaikhare/Documents/Course_work/NLP/NLP_Final_Project/Final-Project-Group3/Code/model/'
# trained_tokenizer = '/Users/shikharaikhare/Documents/Course_work/NLP/NLP_Final_Project/Final-Project-Group3/Code/tokenizer/'

def main():
    parser = argparse.ArgumentParser(description='Train and save model and tokenizer paths')
    parser.add_argument('--model_path', type=str, help='Path to save the model')
    parser.add_argument('--tokenizer_path', type=str, help='Path to save the tokenizer')

    args = parser.parse_args()

    if args.model_path is None or args.tokenizer_path is None:
        print("Please provide paths to the model and tokenizer using --model_path and --tokenizer_path arguments.")
        return

    trained_model_path = args.model_path
    trained_tokenizer = args.tokenizer_path

    model = T5ForConditionalGeneration.from_pretrained(trained_model_path)
    tokenizer = T5Tokenizer.from_pretrained(trained_tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ("device ",device)
    model = model.to(device)

    context ="In Tokyo Olympics, the USA team defeated France 87-82 to win the gold medal"
    answer = "USA team"
    text = "context: "+context + " " + "answer: " + answer + " </s>"
    print (text)

    encoding = tokenizer.encode_plus(text,max_length =512, padding=True, return_tensors="pt")
    print (encoding.keys())
    input_ids,attention_mask  = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    model.eval()
    beam_outputs = model.generate(
        input_ids=input_ids,attention_mask=attention_mask,
        max_length=72,
        early_stopping=True,
        num_beams=5,
        num_return_sequences=3

    )

    print(beam_outputs)
    print('-----------------------------------------------------------------------------------------------')
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        print (sent)

if __name__ == '__main__':
    main()