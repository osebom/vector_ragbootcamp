# Customer Review Retrieval and Analysis with Weaviate and LLMs

## Overview
This Jupyter Notebook implements a **retrieval-augmented system** for **analyzing customer reviews** using **Weaviate as a vector database** and **LLM models for query processing**. The dataset (`Bootcamp_Dataset_CIBC2.csv`) contains customer reviews with metadata such as **date, rating, topic, headline, and sentiment**.

## Features
- **Weaviate Vector Database**: Stores customer reviews as vector embeddings.
- **LangChain & LlamaIndex**: Implements structured retrieval and query engines.
- **Hugging Face Embeddings**: Converts text reviews into vector representations for similarity search.
- **OpenAI-like Chat Models**: Enables interactive querying and review analysis.
- **Sentiment-Based Queries**: Retrieves and analyzes reviews based on sentiment, ratings, and topics.

## Dependencies
Ensure you have the following Python libraries installed:

```bash
pip install weaviate-client langchain langchain-openai llama-index pandas requests numpy faiss-cpu
```

## Setup Instructions
1. **Connect to Weaviate**: Update the `cluster_url` and API key in the notebook:

```python
client = weaviate.connect_to_weaviate_cloud(
    cluster_url="your_weaviate_url",
    auth_credentials=Auth.api_key("your_api_key")
)
```

2. **Set OpenAI API Key**:

```python
OPENAI_API_KEY = 'your_openai_api_key'
GENERATOR_MODEL_NAME = "Meta-Llama-3.1-8B-Instruct"
GENERATOR_BASE_URL = "https://your_openai_base_url"
```

3. **Load and Process the Dataset**: The notebook reads the review dataset and converts each review into **TextNodes** with metadata:

```python
df = pd.read_csv("/path/to/Bootcamp_Dataset_CIBC2.csv")
nodes = [
    TextNode(
        text=row["reviewBody"],
        metadata={key: value for key, value in row.items() if key in metadata_fields and value is not None}
    )
    for _, row in df.iterrows()
]
```

4. **Store Reviews in Weaviate**:

```python
vector_store = WeaviateVectorStore(
    weaviate_client=client, index_name="Reviews_Llamaindex", text_key="text"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)
```

5. **Retrieve and Analyze Reviews**:

```python
retriever = index.as_retriever(similarity_top_k=6)
retrieved_docs = retriever.retrieve("Tell me about the worst reviews")
```

## Example Query
```python
query_engine = RetrieverQueryEngine(retriever=retriever)
result = query_engine.query("Tell me about the worst reviews")
print(f"Result: \n\n{result}")
```

## Future Enhancements
- Expand review analysis with **topic modeling** and **aspect-based sentiment analysis**.
- Deploy the system as a **FastAPI-based REST API**.
- Implement **real-time feedback monitoring**.

## Author
This project is designed for **customer sentiment analysis** using **LLM-based retrieval systems**. Contributions and feedback are welcome!

