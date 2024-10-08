from models.base_embeddings_adapter import BaseBedrockEmbeddingsAdapter
import logging
import json

logger = logging.getLogger('ModelAdapter')


class ModelAdapter(BaseBedrockEmbeddingsAdapter):

    def get_embedding(self, text):
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
        embedding = response_body.get('embeddings').get(embeddings_type[0])[0]
        if len(embeddings_type) > 1:
            logger.warning(f"Multiple embedding types detected. Only the first type will be used: {embeddings_type[0]}")

        return embedding
