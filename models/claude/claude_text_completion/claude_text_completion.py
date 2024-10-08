from models.base_completion_adapter import BaseBedrockCompletionAdapter
import logging
import json

logger = logging.getLogger('ModelAdapter')


class ModelAdapter(BaseBedrockCompletionAdapter):

    def stream_response(self, messages):
        stream = self.configuration.get("stream", True)
        body = {
            "prompt": messages,
            "temperature": self.configuration.get("temperature", 1.0),
            "top_p": self.configuration.get("top_p", 1.0),
            "top_k": self.configuration.get("top_k", 250),
            "max_tokens_to_sample": self.configuration.get("max_tokens", 200),
            "stop_sequences": self.configuration.get("stop_sequences", []),

        }
        body_bytes = json.dumps(body).encode('utf-8')
        if stream:
            streaming_response = self.client.invoke_model_with_response_stream(
                modelId=self.model_id, body=body_bytes
            )

            # Extract and yield the response text in real-time.
            for chunk in streaming_response["body"]:
                completion = chunk.get('chunk').get('bytes').decode('utf-8')
                completion = json.loads(completion)
                yield completion.get('completion') or ""

        else:
            response = self.client.invoke_model(body=body_bytes,
                                                modelId=self.model_id,
                                                contentType='application/json')

            result = response.get('body').read().decode('utf-8')
            result = json.loads(result)
            output_text = result.get("completion")
            yield output_text or ""

    @staticmethod
    def reformat_messages(messages, system_prompt):
        """
        Convert SDK message format to the required format.

        :param system_prompt: system prompt for the model
        :param messages: A list of messages in the OpenAI format (default by SDK).
        :return: inputText for the request.
        """
        conversation_history = ""

        # Iterate through the messages and format the conversation history
        for message in messages:
            role = message['role']
            if role == 'user':
                role = 'Human'
            else:
                role = role.capitalize()

            for content in message['content']:
                if content['type'] == 'text':
                    conversation_history += f"{role}: {content['text']}\n"

        # Build the final inputText
        input_text = f"{system_prompt}\n\n{conversation_history}\n\nAssistant:"

        return input_text
