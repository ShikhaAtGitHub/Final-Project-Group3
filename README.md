# Final-Project-Group3

- To start run data_preprocess.py, this execution will end up saving 'squad_t5_train.csv' and 'squad_t5_val.csv' in Code folder.
- Now that we have generated SQuaD dataset we can start with training our T5 model.
- Run T5_train.py, this execution will save the model and the tokenizer under the model and tokenizer folder.
- For Evaluation run T5_test.py
- Eventually, run streamlit.py to see the model prediction in action.
- python your_script.py --model_path /path/to/save/model --tokenizer_path /path/to/save/tokenizer
- To run streamlit_app.py use the below command:
 streamlit run streamlit_app.py -- --model_path /Users/shikharaikhare/Documents/Course_work/NLP/NLP_Final_Project/Final-Project-Group3/Code/model/ --tokenizer_path /Users/shikharaikhare/Documents/Course_work/NLP/NLP_Final_Project/Final-Project-Group3/Code/tokenizer/
 streamlit run streamlit_app.py -- --model_path path/to/the/model --tokenizer_path path/to/the/tokenizer
- 2
- 

