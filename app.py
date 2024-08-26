from flask import Flask, render_template, request, jsonify, session
import os
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import LLMChain
from langchain import PromptTemplate
from fuzzywuzzy import process

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Set the HuggingFace API token
sec_key = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = sec_key

# Load your custom dataset from CSV with a specified encoding
try:
    custom_data = pd.read_csv('/home/avaxpro/Desktop/patil/data.csv', encoding='latin1')
except UnicodeDecodeError:
    custom_data = pd.read_csv('/home/avaxpro/Desktop/patil/data.csv', encoding='cp1252')

# Create documents for each entry
documents = [Document(page_content=f"Q: {row['Question']} A: {row['Answer']}") for index, row in custom_data.iterrows()]

# Load documents into a Chroma vector store
embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = Chroma.from_documents(documents, embedding=embedding_function)

# Set up the language model
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=sec_key)

# Create a prompt template
prompt_template = PromptTemplate(template="Answer the question based on the provided context:\n{context}\n\nQuestion: {question}\nAnswer:", input_variables=["context", "question"])

# Create an LLMChain
chain = LLMChain(prompt=prompt_template, llm=llm)

def custom_retrieval(question, documents, threshold=80):
    doc_questions = [doc.page_content.split(' A: ')[0].replace('Q: ', '') for doc in documents]
    best_match, score = process.extractOne(question, doc_questions)
    if score < threshold:
        return None
    best_match_index = doc_questions.index(best_match)
    return documents[best_match_index]

def generate_step_by_step_answer(context, start=0, step_size=10):
    context_parts = context.page_content.split('Step')[1:]
    if start >= len(context_parts):
        return "No more steps."
    
    chunk = context_parts[start:start + step_size]
    formatted_steps = [f"Step{step.strip()}" for step in chunk]
    return "<br>".join(formatted_steps)

@app.route('/', methods=['GET', 'POST'])
def index():
    question = ''
    answer = ''
    if request.method == 'POST':
        question = request.form['question']
        context = custom_retrieval(question, documents)
        
        if context:
            session['full_answer'] = context.page_content
            session['last_step'] = 0  # Reset the counter
            answer = generate_step_by_step_answer(context, start=0)
            session['last_step'] += 10  # Initial chunk size
        else:
            answer = "No relevant context found. Call customer care. This is contact 12452435."

    return render_template('index.html', question=question, answer=answer)

@app.route('/api', methods=['POST'])
def api():
    message = request.json.get('message', '')
    context = custom_retrieval(message, documents)
    if context:
        session['full_answer'] = context.page_content
        session['last_step'] = 0  # Reset the counter
        answer = generate_step_by_step_answer(context, start=0)
        session['last_step'] += 10  # Initial chunk size
    else:
        answer = "No relevant context found. Call customer care. This is contact 12452435."

    return jsonify({'message': answer})

@app.route('/api/read_more', methods=['POST'])
def read_more():
    last_step = session.get('last_step', 0)
    context_content = session.get('full_answer', '')
    
    if context_content:
        context = Document(page_content=context_content)
        next_chunk = generate_step_by_step_answer(context, start=last_step)
        session['last_step'] += 10  # Increase the step count for the next chunk
        return jsonify({'message': next_chunk})
    
    return jsonify({'message': "No more steps."})

if __name__ == '__main__':
    app.run(debug=True)
