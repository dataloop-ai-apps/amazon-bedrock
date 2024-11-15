from models.base_embeddings_adapter import BaseBedrockEmbeddingsAdapter
import logging
import json

logger = logging.getLogger('ModelAdapter')


class ModelAdapter(BaseBedrockEmbeddingsAdapter):

    def get_embedding(self, text):
        # The number of dimensions the output embeddings should have. The following values are accepted: 1024 (default), 512, 256.
        dimensions = self.configuration.get("dimensions", 1024)
        # Flag indicating whether to normalize the output embeddings. Defaults to true.
        normalize = self.configuration.get("normalize", True)
        # Accepts a list containing "float", "binary", or both. Defaults to float.
        embeddings_types = [self.configuration.get("embedding_types", "float")]

        if self.model_id == 'amazon.titan-embed-text-v1':
            body = json.dumps({
                "inputText": text,
            })
        elif self.model_id == 'amazon.titan-embed-text-v2:0':
            body = json.dumps({
                "inputText": text,
                "dimensions": dimensions,
                "normalize": normalize,
                "embeddingTypes": embeddings_types

            })
        else:
            raise ValueError("Only Titan Embeddings Text models ids are supported.")

        response = self.client.invoke_model(
            body=body, modelId=self.model_id, accept="application/json", contentType="application/json"
        )

        response_body = json.loads(response.get('body').read())
        embedding = response_body['embedding']

        return embedding
