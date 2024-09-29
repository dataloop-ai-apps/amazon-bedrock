import dtlpy as dl
import logging
import base64
import boto3
import json
import os

logger = logging.getLogger('ModelAdapter')


@dl.Package.decorators.module(description='Model Adapter for Cohere Embeddings',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class ModelAdapter(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        aws_credentials = os.environ.get("AWS_INTEGRATION")
        if aws_credentials is None:
            raise ValueError("Cannot find integrations for AWS")

        decoded_bytes = base64.b64decode(aws_credentials)
        aws_credentials = decoded_bytes.decode("utf-8")
        aws_credentials = json.loads(aws_credentials)
        logger.info("Loaded integrations")

        self.model_id = self.configuration.get("model_id", "cohere.embed-multilingual-v3")
        region = self.configuration.get("region", "eu-central-1")
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

            embeddings_type = self.configuration.get("embedding_types", ["float"])
            body = json.dumps({
                "texts": [text],
                "input_type": self.configuration.get("input_type", 'search_query'),
                # Excepts "search_document", "serac_query", "classification", "clustering".
                "truncate": self.configuration.get("truncate", 'END'),
                # Specifies how the API handles inputs longer than the maximum token length. [NONE, START, END]
                "embedding_types": embeddings_type
                # Specifies the types of embeddings you want to have returned. Accepts a list containing "float", "int", "int8", "binary", "ubinary" or both. Defaults to float.
            })

            response = self.client.invoke_model(
                body=body, modelId=self.model_id, accept="application/json", contentType="application/json"
            )
            response_body = json.loads(response.get('body').read())
            # TODO: Takes the first embeddings type
            embedding = response_body.get('embeddings').get(embeddings_type[0])[0]
            logger.info(f'Extracted embeddings for text {item}: {embedding}')
            embeddings.append(embedding)

        return embeddings


if __name__ == '__main__':
    dl.setenv("rc")
    item = dl.items.get(item_id="66f93ae15fcbd387ba0f5fa4")
    model = dl.models.get(model_id="66de9b36c792b6259b828eba")

    adapter = ModelAdapter(model)
    adapter.embed_items([item])
