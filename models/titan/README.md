# Amazon Titan Models served by Amazon Bedrock in Dataloop Platform

## Overview

Amazon Titan models, offered via Amazon Bedrock, are high-performance machine learning models designed to handle a
variety of NLP tasks such as text generation and embeddings. These models are optimized for scalability and offer
powerful features to help you build intelligent, AI-driven applications.

### 1. Titan Text Models

Titan Text Models are built for open-ended text generation tasks, providing the flexibility to handle complex language
tasks such as content generation, summarization, and conversational AI. With a context length of up to 8,000 tokens,
Titan models excel at capturing and understanding detailed inputs.

#### Key Features:

- **Large Context Window**: Supports up to 8,000 tokens, allowing for complex and detailed language tasks.
- **Advanced Language Generation**: Generates coherent and contextually relevant text based on input prompts.
- **Scalable**: Designed for large-scale deployments through the Amazon Bedrock ecosystem.

#### Use Cases:

- Text summarization
- Content generation
- Conversational agents

### The basic configurations

* ```system_prompt```: Model's system prompt (default: ```You are a helpful and a bit cynical assistant. Give relevant and short answers, if you dont know the answer just say it, dont make up an answer```)
* ```model_id```: Model id. The model ids are from [here](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html).
* ```region```: The region were the models deployed in your AWS account. make sure to edit this field (default: ```""```).
* ```max_tokens```: The maximum number of tokens that the model can generate in a response (default: ```512```).
* ```stream```: Whether to stream the response (default ```true```).
* ```top_p```: Controls the cumulative probability threshold for token selection, similar to nucleus sampling. Default is```0.9```.
* ```temperature```: Adjusts the randomness of the model's output; higher values produce more varied results. Default is ```0.7```.
* ```stop_sequences```: Specifies sequences at which the model will stop generating further tokens. Default is ```[]```.
* 
### 2. Titan Embeddings Models

Titan Embeddings Models are designed to convert text into high-quality embeddings for tasks such as search, clustering,
and recommendation. These models support multilingual embeddings, enabling global-scale applications.

#### Key Features:

- **Multilingual Support**: Handles text in multiple languages, enabling diverse use cases.
- **Dense Vector Representations**: Converts text into embeddings that capture semantic meaning efficiently.
- **Optimized for Search and Clustering**: Ideal for information retrieval, text classification, and recommendation
  systems.

#### Use Cases:

- Semantic search
- Clustering and classification
- Recommendation engines

### The basic configurations

* ```model_id```: Model id. The model ids are from [here](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html).
* ```region```: The region were the models deployed in your AWS account. make sure to edit this field (default: ```["float"]```).

Look [here](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html) for explanations on these variables: 
* ```dimensions```: Default: ```1024```.
* ```normalize```: Default ```true```.
* ```embeddingTypes```: Default ```["float"]```.
