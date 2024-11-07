# Claude Models served by Amazon Bedrock in Dataloop Platform

## Overview

Claude's models are cutting-edge AI models optimized for text completions and multimodal tasks. Served by Amazon
Bedrock,
Claude models are designed for advanced natural language understanding, providing robust solutions for a wide range of
text-based and multimodal applications.


### 1. Claude Text Completions

Claude Text Completions are designed to provide high-quality, human-like completions based on given prompts. The model
excels in understanding context, generating coherent, and contextually accurate responses.

#### Key Features:

- **Contextual Understanding**: Produces relevant and coherent responses based on text prompts.
- **Long-Form Generation**: Ideal for open-ended writing tasks, conversational agents, and text summarization.
- **Customizable**: Can be fine-tuned or parameterized for specific business needs or user intents.

#### Use Cases:

- Conversational AI
- Text summarization
- Automated content generation

### The basic configurations

* ```system_prompt```: Model's system prompt (default: ```You are a helpful and a bit cynical assistant. Give relevant and short answers, if you dont know the answer just say it, dont make up an answer```)
* ```model_id```: Model id. The model ids are from [here](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html).
* ```region```: The region were the models deployed in your AWS account. make sure to edit this field (default: ```""```).
* ```max_tokens```: The maximum number of tokens that the model can generate in a response (default: ```200```).
* ```stream```: Whether to stream the response (default ```true```).
* ```anthropic_version```: Anthropic version (default ```bedrock-2023-05-31```).


### 2. Claude MultiModel Models

Claude MultiModel Models extend the capabilities of Claude text models by incorporating multimodal inputs, such as
images, in addition to text. This allows for enhanced interactions that can understand and respond to complex,
multimodal inputs.

#### Key Features:

- **Multimodal Capabilities**: Handles inputs combining text with other formats like images for more comprehensive
  responses.
- **Seamless Integration**: Works efficiently within the Amazon Bedrock ecosystem, allowing for scalable deployments.
- **Versatile Applications**: Supports diverse use cases like chatbots that interpret both visual and textual data.

#### Use Cases:

- Visual question answering
- Multimodal content analysis
- Enhanced AI assistants capable of understanding both text and images

### The basic configurations

* ```system_prompt```: Model's system prompt (default: ```You are a helpful and a bit cynical assistant. Give relevant and short answers, if you dont know the answer just say it, dont make up an answer```)
* ```model_id```: Model id. The model ids are from [here](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html).
* ```region```: The region were the models deployed in your AWS account. make sure to edit this field (default: ```""```).
* ```max_tokens```: The maximum number of tokens that the model can generate in a response (default: ```200```).
* ```stream```: Whether to stream the response (default ```true```).
* ```top_p```: Controls the cumulative probability threshold for token selection, similar to nucleus sampling. Default is```1.0```.
* ```top_k```: Limits the number of highest-probability tokens considered during generation. Default is ```250```.
* ```temperature```: Adjusts the randomness of the model's output; higher values produce more varied results. Default is ```1.0```.
* ```stop_sequences```: Specifies sequences at which the model will stop generating further tokens. Default is ```[]```.