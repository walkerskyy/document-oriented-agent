from fastapi import FastAPI, HTTPException
from typing import Optional

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pydantic import BaseModel


from dotenv import load_dotenv
import os

load_dotenv()
class Question(BaseModel):
    text: str
    language: str

app = FastAPI()
prompt_template = """Use the following pieces of context to answer the users question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ----------------
    {context}
    ----------------
    Human: {question}
    Answer in HTML format"""

translate_template = """Translate the following text to language: {language}. 
    
    ----------------
    {context}
    ----------------
    Answer in HTML format"""

@app.on_event("startup")
async def startup_event():
    """
    Load all the necessary models and data once the server starts.
    """
    app.directory = '/app/article/'
    app.documents = load_docs(app.directory)
    app.docs = split_docs(app.documents)

    app.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    app.persist_directory = "chroma_db"

    app.vectordb = Chroma.from_documents(
        documents=app.docs,
        embedding=app.embeddings,
        persist_directory=app.persist_directory
    )
    app.vectordb.persist()

    app.model_name = "gpt-3.5-turbo"
    app.llm = ChatOpenAI(model_name=app.model_name)
    
    

    app.prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    app.trans_prompt = PromptTemplate(
        template=translate_template, 
        input_variables=["context", "language"]
    )

    app.db = Chroma.from_documents(app.docs, app.embeddings)
    app.chain = load_qa_chain(
        app.llm, 
        chain_type="stuff", 
        prompt=app.prompt, 
        verbose=True
    )
    app.tr_chain = LLMChain(
        llm=app.llm, 
        prompt=app.trans_prompt, 
        verbose=True
    )


@app.get("/query/")
async def query_chain(question: Question):


    """
    Queries the model with a given question and returns the answer.
    """
    print(question)

    matching_docs_score = app.db.similarity_search_with_score(question.text)

    if len(matching_docs_score) == 0:
        raise HTTPException(status_code=404, detail="No matching documents found")

    matching_docs = [doc for doc, score in matching_docs_score]
    
    answer = app.chain.run(
        input_documents= matching_docs, 
        question= question.text,
        return_only_outputs=True
    )

    tr_answer = app.tr_chain.run(
        context= answer,
        language= question.language,
        return_only_outputs=True
    )
    
    # answer = app.chain.run(input_documents=matching_docs, question=question)

    # Prepare the sources
    sources = [{
        "content": doc.page_content,
        "metadata": doc.metadata,
        "score": score
    } for doc, score in matching_docs_score]

    return {
        "answer": answer, 
        "tr_answer": tr_answer, 
        "sources": sources
    }


def load_docs(directory: str):
    """
    Load documents from the given directory.
    """
    loader = DirectoryLoader(directory)
    documents = loader.load()

    return documents


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    """
    Split the documents into chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)

    return docs
