import dtlpy as dl
import logging
import base64
import boto3
import json
import os

logger = logging.getLogger('ModelAdapter')
aws_access_key_id = ""
aws_secret_access_key = ""
integration_name = "AWS-Integration"


@dl.Package.decorators.module(description='Model Adapter for Amazon Titan Text G1 - Express',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model,
                                           'integration_name': "String"})
class ModelAdapter(dl.BaseModelAdapter):

    def __init__(self, model_entity: dl.Model, integration_name):
        self.integration_name = integration_name
        self.client = None
        self.model_id = 'amazon.titan-text-express-v1'
        super().__init__(model_entity)

    def load(self, local_path, **kwargs):
        aws_credentials = os.environ.get(self.integration_name.replace('-', '_'), None)
        if aws_credentials is None:
            raise ValueError("Cannot find integrations for AWS")

        logger.info(f"Getting integration keys from: {self.integration_name}")

        decoded_bytes = base64.b64decode(aws_credentials)
        aws_credentials = decoded_bytes.decode("utf-8")
        aws_credentials = json.loads(aws_credentials)

        logger.info(f"integration_name = {self.integration_name}")

        self.client = boto3.client(service_name="bedrock-runtime",
                                   region_name='eu-central-1',  # TODO: get region from integration?
                                   aws_access_key_id=aws_credentials['key'],
                                   aws_secret_access_key=aws_credentials['secret'])

    def predict(self, batch, **kwargs):
        annotations = []
        for item in batch:
            prompts = item["prompts"]  # TODO: dl.PromptType
            item_annotations = dl.AnnotationCollection()
            for prompt_key, prompt_content in prompts.items():
                questions = list(prompt_content.values()) if isinstance(prompt_content, dict) else prompt_content
                for i, question in enumerate(questions):
                    if question['mimetype'] == dl.PromptType.TEXT:
                        logger.info(f"User: {question['value']}")

                        body = json.dumps({
                            "inputText": question['value'],  # The prompt
                            "textGenerationConfig": {
                                "maxTokenCount": 4096,
                                # Specify the maximum number of tokens to generate in the response. [215,8000]
                                "stopSequences": [],
                                # Specify a character sequence to indicate where the model should stop.
                                "temperature": 0,  # lower value to decrease randomness in responses [0.0,1.0]
                                "topP": 1
                                # lower value to ignore less probable options and decrease the diversity of responses. (Max:1)
                            }
                        })

                        response = self.client.invoke_model(body=body,
                                                            modelId=self.model_id,
                                                            accept='application/json',
                                                            contentType='application/json')

                        results = json.loads(response.get("body").read())
                        for result in results["results"]:
                            ans = result["outputText"]
                            logger.info("Response: {}".format(ans))
                            item_annotations.add(annotation_definition=dl.FreeText(text=ans),
                                                 prompt_id=prompt_key,
                                                 model_info={'name': self.model_id,
                                                             'confidence': 1.0})  # TODO: Change model's confidence?

                    else:
                        logger.warning(f"Entry ignored. {self.model_id} can only answer to text prompts.")
            annotations.append(item_annotations)
        return annotations

    def prepare_item_func(self, item: dl.Item):
        buffer = json.load(item.download(save_locally=False))
        return buffer

    def train(self, data_path, output_path, **kwargs):
        logger.info("Training not implemented yet")


if __name__ == '__main__':
    dl.setenv("rc")
    project = dl.projects.get(project_name="meta-clip")
    dataset = project.datasets.get(dataset_name="prompts")
    item = dataset.items.get(item_id="66154f18f05be295de0b42d9")
    model = project.models.get(model_id="66154f5039be89c7c8e3a39d")

    adapter = ModelAdapter(model, integration_name=integration_name)
    adapter.predict_items([item])
