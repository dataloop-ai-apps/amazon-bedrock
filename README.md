# Amazon Bedrock

## Introduction

Amazon Bedrock is a cloud-based service that offers high-performance foundation models (FM) from leading AI companies such as
AI21 Labs, Anthropic, Cohere, Meta, Mistral AI, Stability AI and Amazon through a single API.
It provides tools for building generative AI applications and offers easy experimentation and evaluation of different FMs
for your specific needs which can be further customized using techniques like fine-tuning and Retrieval Augmented Generation (RAG).


## Description

This repo is an integration between [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html)
models and [Dataloop](https://dataloop.ai/).

The Applications provide accesses to AWS Bedrock models, using the AWS SDK for Python (Boto3), as Dataloop model.

The proposed models:

* ```Titan Text G1 - Express``` -  Amazon Titan Text Express, with a context length of up to 8,000 tokens, excels in 
advanced language tasks like open-ended text generation and conversational chat.


### The pipeline service settings:
Edit the pipeline service setting and add your AWS integrations.

* ```Init Inputs Value```: 
  * ```integration name``` The name of your integration.
  

* ```Secrets & Integrations```: 
  * ```integration``` Choose your integration from the list. To create a new integration refer to the Data Governance page.

### Installation - Dataloop platform

Install the model from StartLine.

[//]: # (### Create pipeline node)

[//]: # ()
[//]: # ()
[//]: # ()
[//]: # (### Add integration)

[//]: # (Init parameter have to have same name as secrets name.)
