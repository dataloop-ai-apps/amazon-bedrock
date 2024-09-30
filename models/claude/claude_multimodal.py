import dtlpy as dl
import logging
import base64
import boto3
import json
import os

logger = logging.getLogger('ModelAdapter')


@dl.Package.decorators.module(description='Model Adapter for Claude Models',
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

    def stream_response(self, messages):
        system_prompt = self.configuration.get('system_prompt', "")
        stream = self.configuration.get("stream", True)
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.configuration.get("max_tokens", 200),
            "system": system_prompt,
            "messages": messages

        }
        body_bytes = json.dumps(body)
        if stream:
            streaming_response = self.client.invoke_model_with_response_stream(
                modelId=self.model_id, body=body_bytes
            )

            # Extract and yield the response text in real-time.
            for event in streaming_response.get("body"):
                chunk = json.loads(event["chunk"]["bytes"])
                if chunk['type'] == 'content_block_delta':
                    if chunk['delta']['type'] == 'text_delta':
                        yield chunk['delta']['text'] or ""

        else:
            response = self.client.invoke_model(body=body_bytes,
                                                modelId=self.model_id,
                                                contentType='application/json')

            result = response.get("body").read().decode('utf-8')
            result = json.loads(result)
            output_text = result.get("content")[0].get("text")
            yield output_text or ""

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

    @staticmethod
    def reformat_messages(messages, system_prompt):
        """
        Convert SDK message format to the required format.

        :param messages: A list of messages in the OpenAI format (default by SDK).
        :return: inputText for the request.
        """

        # Iterate through the messages and format the conversation history
        for message in messages:
            for m in message["content"]:
                if "image" in m.get("type"):
                    m["type"] = "image"
                    m["image_url"] = {"type": "base64",
                                      "media_type": ModelAdapter.extract_image_mime_type(m["image_url"].get("url")),
                                      "data": m["image_url"].get("url").split(',')[1]}
                    m["source"] = m.pop("image_url")

        return messages

    @staticmethod
    def extract_image_mime_type(base64_string):
        parts = base64_string.split(',')
        if len(parts) > 1:
            type_info = parts[0]
            # Extract the MIME type, which is in the format "data:image/jpeg;base64"
            mime_type = type_info.split(';')[0].replace('data:', '')
            return mime_type
        return None


if __name__ == '__main__':
    dl.setenv("rc")
    item = dl.items.get(item_id="66fa7689fd4c744f0f0c60f1")
    model = dl.models.get(model_id="66f2c541f70f5e358a65f2a6")

    adapter = ModelAdapter(model)
    adapter.predict_items([item])