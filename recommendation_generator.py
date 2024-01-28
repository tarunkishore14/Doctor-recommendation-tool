import os
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

gemini_llm = ChatGoogleGenerativeAI(google_api_key=os.environ["gemini_api_key"], model="gemini-pro", temperature=0.2)

instructor_embeddings = HuggingFaceEmbeddings()
vectordb_file_path = "faiss_index_doc"

#Create vector database into Disk D so that it does not have to be loaded everytime the streamlit is launched
def create_vector_db():
    loader = CSVLoader(file_path="doctors_data_common_specializations.csv")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)

vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)
retriever = vectordb.as_retriever()

prompt_template = """Given the following context and a question, generate an answer based on this context only. In the answer give the exact text from the "Doctor"
section in the source document context along with the Specialization from the "Speciality" section, without making much changes.
Also give the phone number and the address of the doctor.
If the answer is not found in the context, then give the General Practitioner's details Don't try to make up an answer.

CONTEXT: {context}

QUESTION: {question}"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

doctor_name_chain = RetrievalQA.from_chain_type(
    llm = gemini_llm,
    chain_type = "stuff",
    retriever = retriever,
    input_key = "query",
    return_source_documents = True,
    chain_type_kwargs = {"prompt" : PROMPT}
)


def get_doc(symptom):
    specialization = gemini_llm.invoke(f"Given the symptom '{symptom}', what kind of doctor should I visit?").content

    doc_name = doctor_name_chain(specialization)['result']

    return doc_name

if __name__ == "__helper__":
    create_vector_db()
