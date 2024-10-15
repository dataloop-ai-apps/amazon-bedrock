# Cohere Embeddings Models served by Amazon Bedrock in Dataloop Platform

## Overview

Cohere Embeddings Models, integrated into Amazon Bedrock, provide robust capabilities for converting text into
high-quality embeddings. These embeddings are essential for numerous natural language processing (NLP) tasks, such as:

- **Semantic search**
- **Text classification**
- **Information retrieval**
- **Clustering and recommendation systems**

## Key Features

- **Multilingual Support**: Cohere Embeddings models can handle multiple languages, making them ideal for global
  applications.
- **High-Quality Embeddings**: Designed to produce dense vector representations that capture the meaning of text
  effectively.
- **Scalable Integration**: With Amazon Bedrock, you can easily scale and deploy these models across various NLP tasks
  with minimal effort.
- **Fast and Efficient**: Optimized for performance, allowing fast inference times while maintaining the quality of the
  embeddings.

## Example Use Cases

1. **Semantic Search**: Improve search engine capabilities by indexing documents with embeddings for faster and more
   accurate retrieval of relevant information.

2. **Text Similarity**: Use embeddings to calculate the similarity between text documents, enabling clustering and
   categorization of large datasets.

3. **Recommendation Systems**: Implement personalized recommendations by embedding user data and content for more
   accurate suggestions.

### The basic configurations

* ```model_id```: Model id. The model ids are from [here](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html).
* ```region```: The region were the models deployed in your AWS account. make sure to edit this field (default: ```["float"]```).

Look [here](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html) for explanations on these variables: 
* ```input_type```: Default: ```search_query```.
* ```truncate```: Default ```END```.
* ```embedding_types```: Default ```["float"]```.