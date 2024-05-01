import streamlit as st
import argparse
import requests
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer

# trained_model_path = '/Users/shikharaikhare/Documents/Course_work/NLP/NLP_Final_Project/Final-Project-Group3/Code/model/'
# trained_tokenizer = '/Users/shikharaikhare/Documents/Course_work/NLP/NLP_Final_Project/Final-Project-Group3/Code/tokenizer/'

def load_model(trained_model_path, trained_tokenizer):
    model = T5ForConditionalGeneration.from_pretrained(trained_model_path)
    tokenizer = T5Tokenizer.from_pretrained(trained_tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device ", device)

    model = model.to(device)

    return (model, tokenizer)

def main():
    parser = argparse.ArgumentParser(description='Train and save model and tokenizer paths')
    parser.add_argument('--model_path', type=str, help='Path to save the model')
    parser.add_argument('--tokenizer_path', type=str, help='Path to save the tokenizer')

    args = parser.parse_args()

    if args.model_path is None or args.tokenizer_path is None:
        st.error("Please provide paths to the model and tokenizer using --model_path and --tokenizer_path arguments.")
        return

    st.title("Question Generating")

    with st.spinner('Loading Model Into Memory...'):
        model, tokenizer = load_model(args.model_path, args.tokenizer_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device ", device)

    context = st.text_input('Enter your context here...')
    answer = st.text_input('Enter your answer here...')
    text = "context: "+context + " " + "answer: " + answer + " </s>"
    print (text)

    encoding = tokenizer.encode_plus(text,max_length=512, padding=True, return_tensors="pt")
    print (encoding.keys())
    input_ids,attention_mask  = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    if context and answer:
        with st.spinner('Searching for answers...'):
            prediction = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_length=72,
                early_stopping=True,
                num_beams=5,
                num_return_sequences=3
            )
            print(prediction)
            print('-----------------------------------------------------------------------------------------------')
            for beam_output in prediction:
                sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                st.write(sent)

if __name__ == '__main__':
    main()



