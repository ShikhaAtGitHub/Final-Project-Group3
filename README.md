# Final-Project-Group3
##### References: HuggingFace Library, Followed YouTube Video for generating questions, ChatGPT

- Link to download model and tokenizer folder or use the below commands:
 https://drive.google.com/drive/folders/17g_i4aZemoDlHzdVGsS-SrlYUr8AcnkX?usp=sharing, 
 https://drive.google.com/drive/folders/18cXGLbLFhK4IUDCutKjFiq18D0kOIw69?usp=sharing)
- Go inside Code folder and run "python script.py" (Script to generate the dataset and install required packages)
- Go inside Code folder and run "chmod +x download_gdrive_folders.sh"
- Then run "./download_gdrive_folders.sh"
- Once you have downloaded model and tokenizer folder in place, install streamlit before you run the below command.
- To run streamlit.py use the below command:
 streamlit run streamlit.py -- --model_path path/to/the/model --tokenizer_path path/to/the/tokenizer
- Sample command: streamlit run streamlit.py -- --model_path /Users/shikharaikhare/Documents/Course_work/NLP/NLP_Final_Project/Final-Project-Group3/Code/model/ --tokenizer_path /Users/shikharaikhare/Documents/Course_work/NLP/NLP_Final_Project/Final-Project-Group3/Code/tokenizer/
- Provide any context and answer to the field populated, click 'ENTER' twice, you will see the magic. It will generate the best possible questions.

### Introduction. An overview of the project and an outline of the report
- We selected the problem of question answering using the SQuAD (Stanford Question Answering Dataset) dataset. We chose this problem because question generating is a fundamental task in natural language understanding, and SQuAD is a widely used benchmark dataset for this task.
### Description of the data set.
- We will use the SQuAD dataset, which is freely available and widely used for question answering research and benchmarking.
### Description of the NLP model and what kind of algorithm you use. Provide some background information on the development of the algorithm and include necessary equations and figures.
- T5 for conditional generation is a transformer-based model that generates text based on input prompts or conditions. It uses an encoder-decoder architecture and is trained to produce output text conditioned on the input. It's versatile and can be fine-tuned for various text generation tasks, making it widely applicable in natural language processing.
### Experimental setup. Describe how you are going to use the data to train and test the model. Explain how you will implement the model in the chosen framework and how you will judge the performance.
- Once we download the data using data_preprocess.py file, we use it to 
### What kind of hyper-parameters did you search on? (e.g., learning rate)? How will you detect/prevent overfitting and extrapolation?
- batch = 4, learning rate = 0.0001
- 
### Results. Describe the results of your experiments, using figures and tables wherever possible. Include all results (including all figures and tables) in the main body of the report, not in appendices. Provide an explanation of each figure and table that you include. Your discussions in this section will be the most important part of the report.
- Refer Report under branch 'Kalyani'
### Summary and conclusions. Summarize the results you obtained, explain what you have learned, and suggest improvements that could be made in the future.
- Refer Report under branch 'Kalyani'
### References. In addition to references used for background information or for the written portion, you should provide the links to the websites or github repos you borrowed code from.
- HuggingFace Library, Followed YouTube Video for generating questions, ChatGPT

If you would like to generate dataset and run the model step by step
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


