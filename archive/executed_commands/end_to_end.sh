
# Exit immediately if any command fails
set -e

LANGSMITH_PROJECT="longbenchv2"
python indexing.py --input_file corpus/longbenchv2.json --content_field context --chunk_size 512 --overlap 64 --top_k 256
# python generation.py --collection_name longbenchv2 --top_k 5