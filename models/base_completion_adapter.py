import dtlpy as dl
import logging
import base64
import boto3
import json
import os

logger = logging.getLogger('ModelAdapter')


@dl.Package.decorators.module(description='Model Adapter for Amazon Bedrock Completions Models',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class BaseBedrockCompletionAdapter(dl.BaseModelAdapter):

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
            raise ValueError("You must provide region on the model's configuration.")

        self.client = boto3.client(service_name="bedrock-runtime",
                                   region_name=region,
                                   aws_access_key_id=aws_credentials['key'],
                                   aws_secret_access_key=aws_credentials['secret'])

    def predict(self, batch, **kwargs):
        system_prompt = self.configuration.get('system_prompt', "")
        for prompt_item in batch:
            # Get all messages including model annotations
            _messages = prompt_item.to_messages(model_name=self.model_entity.name)

            nearest_items = prompt_item.prompts[-1].metadata.get('nearestItems', [])
            if len(nearest_items) > 0:
                context = prompt_item.build_context(nearest_items=nearest_items,
                                                    add_metadata=self.configuration.get("add_metadata"))
                logger.info(f"Nearest items Context: {context}")
                _messages.append({"role": "assistant", "content": context})

            messages = self.reformat_messages(messages=_messages, system_prompt=system_prompt)
            streamed_response = self.stream_response(messages=messages)
            response = ""
            for chunk in streamed_response:
                #  Build text that includes previous stream
                response += chunk
                prompt_item.add(message={"role": "assistant",
                                         "content": [{"mimetype": dl.PromptType.TEXT,
                                                      "value": response}]},
                                stream=True,
                                model_info={'name': self.model_entity.name,
                                            'confidence': 1.0,
                                            'model_id': self.model_entity.id})

        return []

    def prepare_item_func(self, item: dl.Item):
        prompt_item = dl.PromptItem.from_item(item)
        return prompt_item

    def stream_response(self, messages):
        raise NotImplementedError("Please implement `stream_response` method in {}".format(self.__class__.__name__))

    @staticmethod
    def reformat_messages(messages, system_prompt):
        raise NotImplementedError("Please implement `reformat_messages` static method.")
