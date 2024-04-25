import streamlit as st
import requests
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer

st.title("Question-Answer Generating")

trained_model_path = '/Users/shikharaikhare/Documents/Course_work/NLP/NLP_Final_Project/Final-Project-Group3/Code/model/'
trained_tokenizer = '/Users/shikharaikhare/Documents/Course_work/NLP/NLP_Final_Project/Final-Project-Group3/Code/tokenizer/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device ", device)

def load_model():
    model = T5ForConditionalGeneration.from_pretrained(trained_model_path)
    tokenizer = T5Tokenizer.from_pretrained(trained_tokenizer)

    model = model.to(device)

    return (model, tokenizer)

with st.spinner('Loading Model Into Memory...'):
    model, tokenizer = load_model()

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

