# from flask import Flask, render_template, request, jsonify, session
# import os
# import pandas as pd
# from apscheduler.schedulers.background import BackgroundScheduler
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.docstore.document import Document
# from langchain.chains import LLMChain
# from langchain import PromptTemplate
# from fuzzywuzzy import process
# from io import StringIO
# import requests
# from dotenv import load_dotenv


# load_dotenv()
# # Initialize Flask app
# app = Flask(__name__)
# app.secret_key = os.getenv('FLASK_SECRET_KEY')

# # Set the HuggingFace API token
# sec_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = sec_key


# # https://docs.google.com/spreadsheets/d/1Dt1Zud6fLTiNRx-_r7EaPiHAhQDcfcK5M9xxhX0x--Q/edit?gid=1023326500#gid=1023326500

# # Google Sheet details
# file_id = '1Dt1Zud6fLTiNRx-_r7EaPiHAhQDcfcK5M9xxhX0x--Q'  # Replace with your actual file ID
# sheet_gid = '1023326500'  # Replace with your actual sheet GID
# url = f'https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv&id={file_id}&gid={sheet_gid}'


# # Initialize global variables
# documents = []
# vectorstore = None

# # def fetch_google_sheet_data():
# #     global documents, vectorstore
    
# #     # Fetch the CSV file from Google Drive
# #     response = requests.get(url)
# #     response.raise_for_status()  # Ensure we notice bad responses

# #     # Read the CSV data into a pandas DataFrame
# #     csv_data = StringIO(response.text)
# #     custom_data = pd.read_csv(csv_data)

# #     # Debugging: Print the data loaded from Google Sheet
# #     print("Data loaded from Google Sheet:")
# #     print(custom_data.head())  # Print the first few rows

# #     # Create documents for each entry
# #     documents = [Document(page_content=f"Q: {row['Question']} A: {row['Answer']}") for index, row in custom_data.iterrows()]

# #     # Load documents into a Chroma vector store
# #     embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
# #     vectorstore = Chroma.from_documents(documents, embedding=embedding_function)

# def fetch_google_sheet_data():
#     global documents, vectorstore
#     response = requests.get(url)
#     response.raise_for_status()
    
#     csv_data = StringIO(response.text)
#     custom_data = pd.read_csv(csv_data)

#     # Debugging - Check if data is loaded
#     print("Loaded data from Google Sheets:")
#     print(custom_data.head())
    
#     documents = [Document(page_content=f"Q: {row['Question']} A: {row['Answer']}") for index, row in custom_data.iterrows()]

#     # Debugging - Check if documents are created
#     print(f"Created {len(documents)} documents.")
#     for doc in documents:
#         print(doc.page_content)

#     embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     vectorstore = Chroma.from_documents(documents, embedding=embedding_function)


# print("Documents created:")
# for doc in documents:
#     print(doc.page_content)


# # Initial data load
# fetch_google_sheet_data()

# # Schedule data refresh every 5 minutes
# scheduler = BackgroundScheduler()
# scheduler.add_job(fetch_google_sheet_data, 'interval', minutes=2)   ##refreshhh time
# scheduler.start()

# # Set up the language model
# repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
# llm = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=sec_key)

# # Create a prompt template
# prompt_template = PromptTemplate(template="Answer the question based on the provided context:\n{context}\n\nQuestion: {question}\nAnswer:", input_variables=["context", "question"])

# # Create an LLMChain
# chain = LLMChain(prompt=prompt_template, llm=llm)

# '''
# In this code, the threshold parameter is used to determine how similar the input question must be to 
# the best-matching question found in the list of doc_questions in order to consider it a relevant match.'''

# # def custom_retrieval(question, documents, threshold=80):
# #     doc_questions = [doc.page_content.split(' A: ')[0].replace('Q: ', '') for doc in documents]
# #     best_match, score = process.extractOne(question, doc_questions)
    
# #     print(f"Best match for '{question}': {best_match} with score {score}")  # Debugging
    
# #     if score < threshold:
# #         print("No relevant context found.")  # Debugging
# #         return None
    
# #     best_match_index = doc_questions.index(best_match)
# #     selected_document = documents[best_match_index]
# #     print(f"Selected document: {selected_document.page_content}")  # Debugging
    
# #     return selected_document

# def custom_retrieval(question, documents, threshold=65):
#     doc_questions = [doc.page_content.split(' A: ')[0].replace('Q: ', '') for doc in documents]
#     best_match, score = process.extractOne(question, doc_questions)
    
#     print(f"Best match: {best_match} with score {score}")
    
#     if score < threshold:
#         print("No relevant match found for this question.")
#         return None
    
#     best_match_index = doc_questions.index(best_match)
#     selected_document = documents[best_match_index]
#     print(f"Selected document: {selected_document.page_content}")
    
#     return selected_document



# # def generate_step_by_step_answer(context, start=0, step_size=10):
# #     # Extract the steps based on the keyword "Step"
# #     context_parts = context.page_content.split('Step ')
    
# #     # Remove any empty parts
# #     context_parts = [part for part in context_parts if part.strip()]

# #     # Check if there are any steps left to display
# #     if start >= len(context_parts):
# #         return "No more steps."
    
# #     # Fetch the steps from the specified range
# #     chunk = context_parts[start:start + step_size]
# #     formatted_steps = [f"Step {step.strip()}" for step in chunk]
    
# #     # Join steps with a line break for better display
# #     return "<br>".join(formatted_steps)

# # def generate_step_by_step_answer(context, start=0, step_size=10):
# #     context_parts = context.page_content.split('Step')[1:]
# #     if start >= len(context_parts):
# #         return "No more steps."
    
# #     chunk = context_parts[start:start + step_size]
# #     formatted_steps = [f"Step {step.strip()}" for step in chunk]
# #     return "<br><br>".join(formatted_steps)
       
# # def generate_step_by_step_answer(context, start=0, step_size=10):
# #     context_parts = context.page_content.split('Step')[1:]  # Split on 'Step' and skip the first part if it is empty
# #     if start >= len(context_parts):
# #         return "No more steps."
    
# #     chunk = context_parts[start:start + step_size]  # Get the current chunk of steps
# #     formatted_steps = [f"Step {start + i + 1}: {step.strip()}" for i, step in enumerate(chunk)]  # Format steps properly
    
# #     # Join the steps with a break tag and return them
# #     response = "<br><br>".join(formatted_steps)
    
# #     # If the returned steps cover the whole context, append 'No more steps.'
# #     if start + step_size >= len(context_parts):
# #         response += "<br><br>No more steps."
    
# #     return response

# # def generate_step_by_step_answer(context, start=0, step_size=5):
# def generate_step_by_step_answer(context, start=0, step_size=10):

#     # Split the content into lines, treating each non-empty line as a step
#     steps = [step.strip() for step in context.page_content.split('\n') if step.strip()]
    
#     # Calculate the end index, ensuring we don't go out of bounds
#     end = min(start + step_size, len(steps))
    
#     # Slice the steps
#     chunk = steps[start:end]
    
#     # Join the steps with HTML line breaks
#     formatted_chunk = "<br><br>".join(chunk)
    
#     # Return the formatted chunk and whether there are more steps
#     return formatted_chunk, end < len(steps)



# @app.route('/', methods=['GET', 'POST'])
# def index():
#     question = ''
#     answer = ''
#     if request.method == 'POST':
#         question = request.form['question']
#         context = custom_retrieval(question, documents)
        
#         if context:
#             session['full_answer'] = context.page_content
#             session['last_step'] = 0  # Reset the counter
#             answer = generate_step_by_step_answer(context, start=0)
#             session['last_step'] += 10  # Initial chunk size
#         else:
#             print("No context found for the question.")  # Debugging
#             answer = "No relevant context found. Call customer care. This is contact 12452435."

#     return render_template('index.html', question=question, answer=answer)

# @app.route('/api', methods=['POST'])
# def api():
#     message = request.json.get('message', '')
#     context = custom_retrieval(message, documents)
#     if context:
#         print(f"Context retrieved: {context.page_content}")  # Debugging
#         session['full_answer'] = context.page_content
#         session['last_step'] = 0  # Reset the counter
#         answer = generate_step_by_step_answer(context, start=0)
#         session['last_step'] += 10  # Initial chunk size
#     else:
#         answer = "No relevant context found. Call customer care. This is contact 12452435."
#         print("No context found.")  # Debugging

#     return jsonify({'message': answer})


# @app.route('/api/read_more', methods=['POST'])
# def read_more():
#     last_step = session.get('last_step', 0)
#     context_content = session.get('full_answer', '')
    
#     if context_content:
#         context = Document(page_content=context_content)
#         next_chunk = generate_step_by_step_answer(context, start=last_step)
#         session['last_step'] += 10  # Increase the step count for the next chunk
#         return jsonify({'message': next_chunk})
    
#     return jsonify({'message': "No more steps."})

# if __name__ == '__main__':
#     app.run(debug=True, port=8080)












# from flask import Flask, render_template, request, jsonify, session
# import os
# import pandas as pd
# from apscheduler.schedulers.background import BackgroundScheduler
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.docstore.document import Document
# from langchain.chains import LLMChain
# from langchain_core.prompts import PromptTemplate
# from fuzzywuzzy import process
# from io import StringIO
# import requests
# from dotenv import load_dotenv


# load_dotenv()
# # Initialize Flask app
# app = Flask(__name__)
# app.secret_key = os.getenv('FLASK_SECRET_KEY')

# # Set the HuggingFace API token
# sec_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = sec_key


# # Google Sheet details
# file_id = '1Dt1Zud6fLTiNRx-_r7EaPiHAhQDcfcK5M9xxhX0x--Q'  # Replace with your actual file ID
# sheet_gid = '1023326500'  # Replace with your actual sheet GID
# url = f'https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv&id={file_id}&gid={sheet_gid}'

# documents = []
# vectorstore = None

# def fetch_google_sheet_data():
#     global documents, vectorstore
#     response = requests.get(url)
#     response.raise_for_status()

#     csv_data = StringIO(response.text)
#     custom_data = pd.read_csv(csv_data)

#     # Create documents for each entry
#     documents = [Document(page_content=f"Q: {row['Question']} A: {row['Answer']}") for index, row in custom_data.iterrows()]

#     # Load documents into Chroma vector store
#     embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     vectorstore = Chroma.from_documents(documents, embedding=embedding_function)

# # Initial data load
# fetch_google_sheet_data()

# # Schedule data refresh every 5 minutes
# scheduler = BackgroundScheduler()
# scheduler.add_job(fetch_google_sheet_data, 'interval', minutes=5)
# scheduler.start()

# # Custom retrieval function
# def custom_retrieval(question, documents, threshold=80):
#     doc_questions = [doc.page_content.split(' A: ')[0].replace('Q: ', '') for doc in documents]
#     best_match, score = process.extractOne(question, doc_questions)

#     if score < threshold:
#         return None
    
#     best_match_index = doc_questions.index(best_match)
#     return documents[best_match_index]

# # def generate_step_by_step_answer(context, start=0, step_size=10):
# #     context_parts = context.page_content.split('Step')[1:]
# #     if start >= len(context_parts):
# #         return "No more steps."
    
# #     chunk = context_parts[start:start + step_size]
# #     formatted_steps = [f"Step {start + i + 1}: {step.strip()}" for i, step in enumerate(chunk)]
# #     return "<br><br>".join(formatted_steps)



# # Improved generate_step_by_step_answer function
# def generate_step_by_step_answer(context, start=0, step_size=5):
#     # Split the content into lines, treating each non-empty line as a step
#     steps = [step.strip() for step in context.page_content.split('\n') if step.strip()]
    
#     # Calculate the end index, ensuring we don't go out of bounds
#     end = min(start + step_size, len(steps))
    
#     # Slice the steps
#     chunk = steps[start:end]
    
#     # Join the steps with HTML line breaks
#     formatted_chunk = "<br><br>".join(chunk)
    
#     # Return the formatted chunk and whether there are more steps
#     return formatted_chunk, end < len(steps)

# # Improved API route
# @app.route('/api', methods=['POST'])
# def api():
#     data = request.get_json()
#     question = data.get('message')
    
#     context = custom_retrieval(question, documents)
    
#     if context:
#         answer, has_more = generate_step_by_step_answer(context)
#         session['full_answer'] = context.page_content
#         session['last_step'] = 5  # We've shown the first 5 steps
#         return jsonify({
#             'message': answer,
#             'has_more': has_more
#         })
    
#     return jsonify({
#         'message': "I'm sorry, I couldn't find an answer for that question.",
#         'has_more': False
#     })

# # Improved read_more route
# @app.route('/api/read_more', methods=['POST'])
# def read_more():
#     last_step = session.get('last_step', 0)
#     context_content = session.get('full_answer', '')
    
#     if context_content:
#         context = Document(page_content=context_content)
#         next_chunk, has_more = generate_step_by_step_answer(context, start=last_step)
        
#         if next_chunk:
#             session['last_step'] = last_step + 5  # Increment by 5 steps
#             return jsonify({
#                 'message': next_chunk,
#                 'has_more': has_more
#             })
    
#     return jsonify({
#         'message': "No more steps.",
#         'has_more': False
#     })

# if __name__ == '_main_':
#     app.run(debug=True)


from flask import Flask, render_template, request, jsonify, session
import os
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from fuzzywuzzy import process
from io import StringIO
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY')

# Set the HuggingFace API token
sec_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
os.environ["HUGGINGFACEHUB_API_TOKEN"] = sec_key

# Google Sheet details
file_id = '1Dt1Zud6fLTiNRx-_r7EaPiHAhQDcfcK5M9xxhX0x--Q'  # Replace with your actual file ID
sheet_gid = '1023326500'  # Replace with your actual sheet GID
url = f'https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv&id={file_id}&gid={sheet_gid}'

documents = []
vectorstore = None

# Function to fetch Google Sheets data
def fetch_google_sheet_data():
    global documents, vectorstore
    response = requests.get(url)
    response.raise_for_status()

    csv_data = StringIO(response.text)
    custom_data = pd.read_csv(csv_data)

    # Create documents for each entry
    documents = [Document(page_content=f"Q: {row['Question']} A: {row['Answer']}") for index, row in custom_data.iterrows()]

    # Load documents into Chroma vector store
    embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = Chroma.from_documents(documents, embedding=embedding_function)

# Initial data load
fetch_google_sheet_data()

# Schedule data refresh every 5 minutes
scheduler = BackgroundScheduler()
scheduler.add_job(fetch_google_sheet_data, 'interval', minutes=3)
scheduler.start()

# Custom retrieval function
def custom_retrieval(question, documents, threshold=80):
    doc_questions = [doc.page_content.split(' A: ')[0].replace('Q: ', '') for doc in documents]
    best_match, score = process.extractOne(question, doc_questions)

    if score < threshold:
        return None
    
    best_match_index = doc_questions.index(best_match)
    return documents[best_match_index]

# Improved generate_step_by_step_answer function
def generate_step_by_step_answer(context, start=0, step_size=5):
    steps = [step.strip() for step in context.page_content.split('\n') if step.strip()]
    end = min(start + step_size, len(steps))
    chunk = steps[start:end]
    formatted_chunk = "<br><br>".join(chunk)
    return formatted_chunk, end < len(steps)

@app.route('/')
def home():
    return render_template('index.html')

# API route for answering questions
@app.route('/api', methods=['POST'])
def api():
    data = request.get_json()
    question = data.get('message')
    
    context = custom_retrieval(question, documents)
    
    if context:
        answer, has_more = generate_step_by_step_answer(context)
        session['full_answer'] = context.page_content
        session['last_step'] = 5  # Show the first 5 steps
        return jsonify({
            'message': answer,
            'has_more': has_more
        })
    
    return jsonify({
        'message': "I'm sorry, I couldn't find an answer for that question.",
        'has_more': False
    })

# API route for reading more steps
@app.route('/api/read_more', methods=['POST'])
def read_more():
    last_step = session.get('last_step', 0)
    context_content = session.get('full_answer', '')
    
    if context_content:
        context = Document(page_content=context_content)
        next_chunk, has_more = generate_step_by_step_answer(context, start=last_step)
        
        if next_chunk:
            session['last_step'] = last_step + 5  # Increment by 5 steps
            return jsonify({
                'message': next_chunk,
                'has_more': has_more
            })
    
    return jsonify({
        'message': "No more steps.",
        'has_more': False
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=9090)


