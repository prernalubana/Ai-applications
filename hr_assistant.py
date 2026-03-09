from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
import os

documents = []

dataset_path = "hr_policies_dataset"

for file in os.listdir(dataset_path):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(dataset_path, file))
        documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)
embeddings = OllamaEmbeddings(model="nomic-embed-text")


vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOllama(model="phi3:mini")

print("AI HR Assistant")
print("Ask HR policy questions (type 'exit' to quit)")


while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Retrieve relevant policy chunks
    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an HR assistant helping employees understand company policies.

Use the HR policy context below to answer the question accurately.

HR Policy Context:
{context}

Employee Question:
{question}

Answer clearly based only on the policy information.
"""

    response = llm.invoke(prompt)

    print("\nAssistant:", response.content)