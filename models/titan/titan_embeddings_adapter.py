import dtlpy as dl
import logging
import base64
import boto3
import json
import os

logger = logging.getLogger('ModelAdapter')


@dl.Package.decorators.module(description='Model Adapter for Amazon Titan Embeddings',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model,
                                           'integration_name': "String"})
class ModelAdapter(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        aws_credentials = os.environ.get("AWS_INTEGRATION")
        if aws_credentials is None:
            raise ValueError("Cannot find integrations for AWS")

        decoded_bytes = base64.b64decode(aws_credentials)
        aws_credentials = decoded_bytes.decode("utf-8")
        aws_credentials = json.loads(aws_credentials)
        logger.info("Loaded integrations")

        self.model_id = self.configuration.get("model_id")
        region = self.configuration.get("region")
        if region is "":
            raise ValueError("You must provide integrations on the model's configuration.")

        self.client = boto3.client(service_name="bedrock-runtime",
                                   region_name=region,
                                   aws_access_key_id=aws_credentials['key'],
                                   aws_secret_access_key=aws_credentials['secret'])

    def embed(self, batch, **kwargs):
        embeddings = []
        for item in batch:
            if isinstance(item, str):
                self.adapter_defaults.upload_features = True
                text = item
            else:
                self.adapter_defaults.upload_features = False
                try:
                    prompt_item = dl.PromptItem.from_item(item)
                    is_hyde = item.metadata.get('prompt', dict()).get('is_hyde', False)
                    if is_hyde is True:
                        messages = prompt_item.to_messages(model_name=self.configuration.get('hyde_model_name'))[-1]
                        if messages['role'] == 'assistant':
                            text = messages['content'][-1]['text']
                        else:
                            raise ValueError(f'Only assistant messages are supported for hyde model')
                    else:
                        messages = prompt_item.to_messages(include_assistant=False)[-1]
                        text = messages['content'][-1]['text']

                except ValueError as e:
                    raise ValueError(f'Only mimetype text or prompt items are supported {e}')

            if self.model_id == 'amazon.titan-embed-text-v1':
                body = json.dumps({
                    "inputText": text,
                })
            elif self.model_id == 'amazon.titan-embed-text-v2:0':
                body = json.dumps({
                    "inputText": text,
                    "dimensions": self.configuration.get("dimensions", 1024),
                    # The number of dimensions the output embeddings should have. The following values are accepted: 1024 (default), 512, 256.
                    "normalize": self.configuration.get("normalize", True),
                    # Flag indicating whether or not to normalize the output embeddings. Defaults to true.
                    "embeddingTypes": self.configuration.get("embedding_types", ["float"])
                    # Accepts a list containing "float", "binary", or both. Defaults to float.
                })
            else:
                raise ValueError("Only Titan Embeddings Text models ids are supported.")

            response = self.client.invoke_model(
                body=body, modelId=self.model_id, accept="application/json", contentType="application/json"
            )
            response_body = json.loads(response.get('body').read())
            embedding = response_body['embedding']
            logger.info(f'Extracted embeddings for text {item}: {embedding}')
            embeddings.append(embedding)

        return embeddings
