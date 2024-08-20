from flask import Flask, render_template, request, jsonify
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

# Create a text splitter to handle large texts (if needed)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# Load documents into a Chroma vector store
embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = Chroma.from_documents(documents, embedding=embedding_function)

# Create a retriever
retriever = vectorstore.as_retriever()

# Set up the language model
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=sec_key)

# Create a prompt template
prompt_template = PromptTemplate(template="Answer the question based on the provided context:\n{context}\n\nQuestion: {question}\nAnswer:", input_variables=["context", "question"])

# Create an LLMChain
chain = LLMChain(prompt=prompt_template, llm=llm)

# Custom retrieval function with fuzzy matching
def custom_retrieval(question, documents, threshold=80):
    # Extract questions from documents
    doc_questions = [doc.page_content.split(' A: ')[0].replace('Q: ', '') for doc in documents]

    # Perform fuzzy matching
    best_match, score = process.extractOne(question, doc_questions)

    if score < threshold:
        return None

    best_match_index = doc_questions.index(best_match)
    return documents[best_match_index]

'''def generate_step_by_step_answer(question, context):
    if context:
        # Split context into steps if applicable
        context_parts = context.page_content.split(' A: ')[1].strip().split('\n')
        # Clean up the context parts to remove any existing numbering
        steps = [f"Step {i + 1}: {part.strip().lstrip('1234567890. ')}" for i, part in enumerate(context_parts)]
        answer = "<br>".join(steps)  # Use HTML <br> for line breaks
        return answer
    return "No relevant context found. Call customer care. This is contact 12452435."'''

# def generate_step_by_step_answer(question, context):
#     if context:
#         # Split context into steps if applicable
#         context_parts = context.page_content.split(' A: ')[1].strip().split('\n')
#         # Clean up the context parts to remove any existing numbering and extra whitespace
#         cleaned_steps = []
#         for part in context_parts:
#             # Remove the existing "Step X:" part if present
#             if part.strip().lower().startswith('step'):
#                 part = part.split(':', 1)[1].strip()
#             cleaned_steps.append(part)
        
#         # Prepend the correct step numbers
#         steps = [f"Step {i + 1}: {part}" for i, part in enumerate(cleaned_steps)]
#         answer = "<br>".join(steps)  # Use HTML <br> for line breaks
#         return answer
#     return "No relevant context found. Call customer care. This is contact 12452435."

import re

def generate_step_by_step_answer(question, context):
    if context:
        # Split context into parts based on 'Step' keyword
        context_parts = context.page_content.split('Step')[1:]

        formatted_steps = []
        for index, part in enumerate(context_parts):
            part = part.strip()
            if part:
                # Separate the main step and any potential sub-steps
                split_parts = part.split('\n')
                main_step = split_parts[0].strip()
                sub_steps = split_parts[1:]

                # Format the main step (preserve original numbering)
                formatted_main_step = f"Step {main_step}"

                # Format sub-steps with proper indentation
                formatted_sub_steps = []
                for sub_step in sub_steps:
                    sub_step = sub_step.strip()
                    if sub_step:
                        # Check if the sub-step starts with a lettered list
                        if sub_step[0].lower() in ['a', 'b', 'c', 'd', 'e', 'f', 'g'] and sub_step[1] == '.':
                            formatted_sub_steps.append(f"&nbsp;&nbsp;&nbsp;&nbsp;{sub_step}")
                        else:
                            formatted_sub_steps.append(f"&nbsp;&nbsp;&nbsp;&nbsp;{sub_step}")

                # Combine the main step with its sub-steps
                if formatted_sub_steps:
                    formatted_steps.append(formatted_main_step + "<br>" + "<br>".join(formatted_sub_steps))
                else:
                    formatted_steps.append(formatted_main_step)
        
        # Combine all steps into a single HTML string
        answer = "<br>".join(formatted_steps)
        return answer
    
    return "No relevant context found. Call customer care. This is contact 12452435."


'''test code :
def generate_step_by_step_answer(question, context):
    if context:
        # Split context into steps if applicable
        context_parts = context.page_content.split(' A: ')[1].strip().split('\n')
        print("Original context parts:", context_parts)  # Debug line
        
        # Clean up the context parts to remove any existing numbering and extra whitespace
        cleaned_steps = []
        for part in context_parts:
            # Remove the existing "Step X:" part if present
            if part.strip().lower().startswith('step'):
                part = part.split(':', 1)[1].strip()
            cleaned_steps.append(part)
        print("Cleaned steps:", cleaned_steps)  # Debug line
        
        # Prepend the correct step numbers
        steps = [f"Step {i + 1}: {part}" for i, part in enumerate(cleaned_steps)]
        answer = "<br>".join(steps)  # Use HTML <br> for line breaks
        print("Final steps:", steps)  # Debug line
        
        return answer
    return "No relevant context found. Call customer care. This is contact 12452435."
'''


@app.route('/', methods=['GET', 'POST'])
def index():
    question = ''
    answer = ''
    if request.method == 'POST':
        question = request.form['question']
        context = custom_retrieval(question, documents)
        answer = generate_step_by_step_answer(question, context)
    return render_template('index.html', question=question, answer=answer)

@app.route('/api', methods=['POST'])
def api():
    data = request.get_json()
    question = data.get('message')
    context = custom_retrieval(question, documents)
    answer = generate_step_by_step_answer(question, context)
    return jsonify({'message': answer})

if __name__ == '__main__':
    app.run(debug=True)
