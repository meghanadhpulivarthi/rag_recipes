from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from dotenv import load_dotenv
from langchain_milvus import Milvus
from langchain import hub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
import logging
import argparse
from operator import itemgetter
from IPython.core.debugger import Pdb
import torch
import traceback

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_collection(collection_name):
    print(f"\n🔄 Starting Collection Load\n- Collection Name: {collection_name}")
    
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    
    print(f"⚡ Embedding Model Loaded: {model_name}")
    
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": "db/milvus.db"},
        collection_name=collection_name,
    )

    del embeddings
    
    print(f"🔍 Collection Loaded: {collection_name}")
    print(f"- Connection URI: db/milvus.db")
    print(f"- Embedding Model: {model_name}\n")
    return vectorstore

def get_retriever(vectorstore, top_k=5):
    print(f"\n🔄 Configuring Retriever\n- Top K: {top_k}")
    print(f"- Search Type: Vector Similarity")
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    print(f"🎯 Retriever Configured\n- Top K: {top_k}")
    print(f"- Search Parameters: {retriever.search_kwargs}\n")
    return retriever




def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def parse_args():
    parser = argparse.ArgumentParser(description='Document chunking utility')
    parser.add_argument('--collection_name', type=str, default="demo",
                        help='Name of the collection (default: demo)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of documents to retrieve (default: 5)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"\n🔄 Starting RAG Pipeline\n- Collection: {args.collection_name}")
    print(f"- Top K: {args.top_k}")
    
    # Load collection
    vectorstore = load_collection(args.collection_name)
    retriever = get_retriever(vectorstore, top_k=args.top_k)

    # Prompt
    print("\n📝 Loading Prompt Template")
    prompt_template = """Please read the following retrieved text chunks and answer the question below.

<text>
{context}
</text>

What is the correct answer to this question: {question}
Choices:
(A) {C_A}
(B) {C_B}
(C) {C_C}
(D) {C_D}

Format your response as follows: "The correct answer is (insert answer here)".
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)


    # LLM
    print("\n🤖 Loading Language Model")
    model_id = "Qwen/Qwen2.5-7B"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, do_sample=False, return_full_text=False)

    llm = HuggingFacePipeline(pipeline=pipe)

    print("\n🔗 Building RAG Chain")
    context_chain = (itemgetter("question") | retriever | format_docs)
    rag_chain = (
        {"context": context_chain, "question": itemgetter("question"), "C_A": itemgetter("C_A"), "C_B": itemgetter("C_B"), "C_C": itemgetter("C_C"), "C_D": itemgetter("C_D")}
        | prompt
        | llm
        | StrOutputParser()
        | (lambda x: x.replace("The correct answer is ", ""))
    )

    # Prepare and run query
    print("\n🔍 Running Query")
    query = "Narrives: 1. The author reflects on leaving the legal profession and transitioning to a more fulfilling career, emphasizing the importance of mentorship and strategic thinking, which proved valuable during their time in the White House.\n2. Barack and Michelle Obama, during their time in the White House, expanded public tours and increased the size of the annual Easter Egg Roll as a way to promote openness and inclusivity. \n3. Sasha Obama's encounter with Chewbacca at a Halloween party scared her so much that she retreated to her bedroom until reassured he had left.\n4. Michelle reflects on the moment she first met Barack, recalling his charming smile and slight awkwardness as he arrived late to his first day at the law firm, not yet realizing he would become her greatest love.\nQuery: Considering the given book and narratives, Which order of the narratives in the following options is correct?"
    choice_A = "3142"
    choice_B = "1423"
    choice_C = "4321"
    choice_D = "3241"
    
    try:
        result = rag_chain.invoke({"question": query, "C_A": choice_A, "C_B": choice_B, "C_C": choice_C, "C_D": choice_D})
        print(f"\n✅ Query Result:\n{result}")
    except Exception as e:
        print("❌ Exception occurred:")
        traceback.print_exc()
    
    print("\n✅ Process Complete!")
