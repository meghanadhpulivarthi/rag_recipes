from langchain_milvus import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_text_splitters import TokenTextSplitter
import argparse
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
import time
from operator import itemgetter
import os
import sys
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/indexing.log')
    ]
)
logger = logging.getLogger(__name__)

def log_info(message):
    """Log info message and flush output"""
    logger.info(message)
    print(f"{datetime.now().isoformat()} - {message}", flush=True)

def log_error(message):
    """Log error message and flush output"""
    logger.error(message)
    print(f"{datetime.now().isoformat()} - ERROR: {message}", file=sys.stderr, flush=True)

load_dotenv()

model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
log_info(f"Embedding Model Loaded: {model_name}")

def fixed_chunk_docs(docs, chunk_size=1000, overlap=200):
    log_info(f"Starting Document Chunking")
    log_info(f"- Total Documents: {len(docs)}")
    log_info(f"- Chunk Size: {chunk_size} tokens")
    log_info(f"- Chunk Overlap: {overlap} tokens")
    
    splitter = TokenTextSplitter(disallowed_special=(), chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(docs)
    log_info(f"Document Chunking Complete")
    log_info(f"- Total Chunks: {len(chunks)}")
    log_info(f"- Average Chunk Length: {sum(len(chunk.page_content) for chunk in chunks)/len(chunks):.2f} characters")
    log_info(f"- Example Chunk: {chunks[0].page_content[:100]}...")
    return chunks

def create_index(docs, collection_name):
    log_info(f"Starting Index Creation")
    log_info(f"- Collection Name: {collection_name}")
    log_info(f"- Total Chunks to Index: {len(docs)}")
    
    start_time = time.time()
    vectorstore = Milvus.from_documents(
        documents=docs,
        embedding=embeddings,
        connection_args={
            "uri": "/dccstor/meghanadhp/projects/myRAG/db/milvus.db",
        },
        collection_name=collection_name,
        drop_old=True,  # Drop the old Milvus collection if it exists
    )
    end_time = time.time()
    log_info(f"✅ Index Created: {collection_name}")
    log_info(f"⏱️ Indexing Time: {end_time - start_time:.2f} seconds")
    log_info(f"- Chunks per Second: {len(docs)/(end_time - start_time):.2f}\n")

    return vectorstore

def get_retriever(vectorstore, top_k=5):
    log_info(f"Configuring Retriever")
    log_info(f"- Top K: {top_k}")
    log_info(f"- Search Type: Vector Similarity")
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    log_info(f"🎯 Retriever Configured")
    log_info(f"- Top K: {top_k}")
    log_info(f"- Search Parameters: {retriever.search_kwargs}\n")
    return retriever

def format_docs(docs):
    return [{"content": doc.page_content, "c_idx": idx} for idx, doc in enumerate(docs)]


def parse_args():
    parser = argparse.ArgumentParser(description='Document chunking utility')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to input JSONL file')
    parser.add_argument('--content_field', type=str, required=True,
                        help='Name of the field containing document content')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='Size of each chunk (default: 1000)')
    parser.add_argument('--overlap', type=int, default=200,
                        help='Overlap between chunks (default: 200)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of documents to retrieve (default: 5)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Load documents
    log_info(f"Starting Document Loading")
    log_info(f"- Input File: {args.input_file}")
    log_info(f"- Content Field: {args.content_field}")
    
    # Read input data
    corpus_df = pd.read_json(args.input_file)
    log_info(f"- Total Rows in JSON: {len(corpus_df)}")
    
    # Initialize output file with empty list if it doesn't exist
    output_file = "/dccstor/meghanadhp/projects/LongBench/data.json"
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            json.dump([], f)
    
    # Load existing results to resume progress
    try:
        with open(output_file, 'r') as f:
            existing_results = json.load(f)
        processed_ids = {item['_id'] for item in existing_results}
        log_info(f"- Found {len(existing_results)} previously processed documents")
    except (json.JSONDecodeError, FileNotFoundError):
        existing_results = []
        processed_ids = set()
    
    for idx, row in corpus_df.iterrows():
        # Skip already processed documents
        if row['_id'] in processed_ids:
            log_info(f"Skipping already processed document: {row['_id']}")
            continue
            
        # Create metadata by including all fields except content_field
        timestamp = time.time()
        metadata = row.to_dict()
        metadata.pop(args.content_field, None)  # Remove the content field
        doc = Document(page_content=row[args.content_field], metadata=metadata)
        log_info(f"Processing document {idx+1}/{len(corpus_df)} - ID: {row['_id']}")
        log_info(f"- Document Metadata: {list(doc.metadata.keys())}")

        try:
            # Chunk documents
            log_info(f"Chunking documents with size={args.chunk_size}, overlap={args.overlap}")
            chunks = fixed_chunk_docs([doc], chunk_size=args.chunk_size, overlap=args.overlap)
            
            # Index chunks
            log_info("Indexing chunks...")
            vectorstore = create_index(chunks, "_" + row['_id'])
            log_info("✅ Indexing complete")
            
            top_k = min(args.top_k, len(chunks))
            log_info(f"- Top K: {top_k}")
            retriever = get_retriever(vectorstore, top_k=min(args.top_k, len(chunks)))

            log_info("Building Context Chain")
            context_chain = (itemgetter("question") | retriever | format_docs)
            retrieved_context = context_chain.invoke({"question": row['question']})
            
            # Create result entry
            result = row.to_dict()
            result['retrieved_context'] = retrieved_context
            result['num_retrieved_chunks'] = len(retrieved_context)
            
            # Append to existing results
            existing_results.append(result)
            
            # Save after each document
            with open(output_file, 'w') as f:
                json.dump(existing_results, f, indent=2)
                
            log_info(f"✅ Saved results for document {row['_id']}")
            log_info(f"⏱️  Processing Time: {time.time() - timestamp:.2f} seconds")
            
        except Exception as e:
            log_error(f"Error processing document {row['_id']}: {str(e)}")
            import traceback
            log_error(traceback.format_exc())
            continue
