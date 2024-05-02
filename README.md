# Final-Project-Group3
##### References: HuggingFace Library, Followed YouTube Video for generating questions, ChatGPT

(Instead of running the T5_train.py file, download model and tokenizer folder and replace with the existing one: https://drive.google.com/drive/folders/17g_i4aZemoDlHzdVGsS-SrlYUr8AcnkX?usp=sharing, https://drive.google.com/drive/folders/18cXGLbLFhK4IUDCutKjFiq18D0kOIw69?usp=sharing)
- https://drive.google.com/drive/folders/18cXGLbLFhK4IUDCutKjFiq18D0kOIw69?usp=sharing
- pip install datasets==1.0.2
- pip install tqdm==4.57.0
- pip install scikit-learn
- The dataset will be generated after sometime when you execute: python data_preprocess.py
- pip install --quiet transformers==4.28.1
- If you are facing issue while installing sentencepiece use this command: conda install -c conda-forge sentencepiece
- pip install torch
- pip install pytorch-lightning
- Once you have replaced model and tokenizer folder in place (), install streamlit before you run the below command.
- To run streamlit.py use the below command:
 streamlit run streamlit.py -- --model_path path/to/the/model --tokenizer_path path/to/the/tokenizer
- Provide any context and answer to the field populated, click 'ENTER' twice, you will see the magic. It will generate the best possible questions.

If you would like to generate dataset and run the model
- pip install datasets==1.0.2
- pip install tqdm==4.57.0
- pip install scikit-learn
- The dataset will be generated after sometime when you execute: python data_preprocess.py
- pip install --quiet transformers==4.28.1
- If you are facing issue installing sentencepiece use this command: conda install -c conda-forge sentencepiece
- pip install torch
- pip install pytorch-lightning
- (Instead of running the T5_train.py file, download model and tokenizer folder and replace with the existing one)Run T5_train.py, this execution will save the model and the tokenizer under the model and tokenizer folder.
- Once you have model and tokenizer folder in place to evaluate run "python T5.test.py"
- To run streamlit.py use the below command:
 streamlit run streamlit.py -- --model_path path/to/the/model --tokenizer_path path/to/the/tokenizer
- Provide any context and answer to it, click 'ENTER' twice, you will see the magic. It will generate the best possible questions.


