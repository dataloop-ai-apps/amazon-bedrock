from models.base_completion_adapter import BaseBedrockCompletionAdapter
import logging
import json

logger = logging.getLogger('ModelAdapter')


class ModelAdapter(BaseBedrockCompletionAdapter):

    def stream_response(self, messages):
        stream = self.configuration.get("stream", True)
        temperature = self.configuration.get("temperature", 0.7)
        top_p = self.configuration.get("top_p", 1.0)
        max_token_count = self.configuration.get("max_tokens", 4096)
        stop_sequences = self.configuration.get("stop_sequences", [])

        body = {
            "inputText": messages,
            "textGenerationConfig": {
                "temperature": temperature,
                "topP": top_p,
                "maxTokenCount": max_token_count,
                "stopSequences": stop_sequences,
            }
        }
        body_bytes = json.dumps(body).encode('utf-8')

        if stream:
            streaming_response = self.client.invoke_model_with_response_stream(
                modelId=self.model_id, body=body_bytes
            )

            # Extract and yield the response text in real-time.
            for event in streaming_response["body"]:
                chunk = json.loads(event["chunk"]["bytes"])
                if "outputText" in chunk:
                    yield chunk["outputText"] or ""

        else:
            response = self.client.invoke_model(body=body_bytes,
                                                modelId=self.model_id,
                                                contentType='application/json')

            result = response['body'].read().decode('utf-8')
            result = json.loads(result)
            output_text = result["results"][0]["outputText"]
            yield output_text

    @staticmethod
    def reformat_messages(messages, system_prompt):
        """
        Convert SDK message format to the required format.

        :param messages: A list of messages in the OpenAI format (default by SDK).
        :return: inputText for the request.
        """
        conversation_history = ""

        # Iterate through the messages and format the conversation history
        for message in messages:
            role = message['role'].capitalize()  # Capitalize 'user' and 'assistant'
            for content in message['content']:
                if content['type'] == 'text':
                    conversation_history += f"{role}: {content['text']}\n"

        # Build the final inputText
        input_text = f"{system_prompt}\n{conversation_history}Bot: "

        return input_text
