import streamlit as st
from test import *

def main():
    st.title('Question Generation and Tool Kit')

    # Inputs
    context = st.text_area("Context:", "Enter some context here...")
    answer = st.text_input("Answer:", "Enter an answer here...")

    # Button to generate results
    if st.button('Generate'):
        results = process_input(context, answer)
        for result in results:
            st.write("Generated Sentence:", result['sentence'])
            st.write("Distractors:", ', '.join(result['distractors']))

if __name__ == '__main__':
    main()


