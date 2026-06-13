import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

def main():
    # Double-check that the API key is loaded
    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable not found.")
        print("Please set it using: export GOOGLE_API_KEY='your_key'")
        return

    print("Step 1: Loading 'company_policy.txt'...")
    try:
        loader = TextLoader("company_policy.txt")
        documents = loader.load()
    except FileNotFoundError:
        print("ERROR: 'company_policy.txt' not found in this directory. Please create it first.")
        return

    print("Step 2: Splitting text into chunks...")
    # Standard configuration: 1000 characters per chunk, 200 overlapping characters
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    print("Step 3: Initializing Gemini Embeddings and Vector Database...")
    # We use the standard Google embedding model for RAG text tasks
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    
    # This creates a local directory named 'chroma_db' to store the vectors
    vector_store = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory="./chroma_db"
    )
    
    # Configure the database to work as a retriever pulling top 2 relevant blocks
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    print("Step 4: Setting up Gemini LLM and RAG Pipeline...")
    # Gemini 2.5 Flash is highly optimized, fast, and cost-effective for RAG
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # Tight instructions ensuring the bot uses your data instead of guessing
    system_prompt = (
        "You are a helpful company HR assistant.\n"
        "Answer the user's question using ONLY the provided context below. "
        "If you do not know the answer or if it's not in the context, "
        "say 'I cannot find that information in the company policy.'\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Tie the LLM, Prompt, and Retriever together
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\n--- RAG System Ready ---")
    
    # Test Question 1 (Information is in the text)
    query_1 = "How much stipend do engineers get for setting up their home office?"
    print(\nf"User Question: '{query_1}'")
    response_1 = rag_chain.invoke({"input": query_1})
    print(f"Gemini Answer: {response_1['answer']}")

    # Test Question 2 (Information is NOT in the text to test hallucination defense)
    query_2 = "What is the policy for parental leave?"
    print(f"\nUser Question: '{query_2}'")
    response_2 = rag_chain.invoke({"input": query_2})
    print(f"Gemini Answer: {response_2['answer']}")

if __name__ == "__main__":
    main()