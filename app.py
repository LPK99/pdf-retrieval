import streamlit as st
from function import load_document, load_model
from langchain.chains import RetrievalQA
def main():
    st.title("Ask your PDF!")
    uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
    if uploaded_file is not None:
        retriever = load_document(uploaded_file)
    input = st.text_area("Enter your question")
    if st.button("Submit your question"):
        llm = load_model(device_type='cpu', model_id='TheBloke/Llama-2-7B-Chat-GGML', model_basename='llama-2-7b-chat.ggmlv3.q4_0.bin')
        rqa = RetrievalQA.from_chain_type(llm=llm, 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)
        answer = rqa(input)['result']
        st.write(answer)
    
if __name__ == "__main__":
    main()