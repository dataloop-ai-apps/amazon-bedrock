from models.base_completion_adapter import BaseBedrockCompletionAdapter
import logging
import json

logger = logging.getLogger('ModelAdapter')


class ModelAdapter(BaseBedrockCompletionAdapter):

    def stream_response(self, messages):
        system_prompt = self.configuration.get('system_prompt', "")
        stream = self.configuration.get("stream", True)
        anthropic_version = self.configuration.get("anthropic_version", "bedrock-2023-05-31")
        max_tokens = self.configuration.get("max_tokens", 200)

        body = {
            "anthropic_version": anthropic_version,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
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

    @staticmethod
    def reformat_messages(messages, system_prompt):
        """
        Convert SDK message format to the required format.

        :param messages: A list of messages in the OpenAI format (default by SDK).
        :return: inputText for the request.
        """

        # Iterate through the messages and format the conversation history
        for message in messages:
            for m in message.get("content"):
                if "image" in m.get("type"):
                    m["type"] = "image"
                    m["image_url"] = {"type": "base64",
                                      "media_type": ModelAdapter.extract_image_mimetype(m["image_url"].get("url")),
                                      "data": m["image_url"].get("url").split(',')[1]}
                    m["source"] = m.pop("image_url")

        return messages

    @staticmethod
    def extract_image_mimetype(base64_string):
        parts = base64_string.split(',')
        if len(parts) > 1:
            type_info = parts[0]
            # Extract the MIME type, which is in the format "data:image/jpeg;base64"
            mimetype = type_info.split(';')[0].replace('data:', '')
            return mimetype
        return None
