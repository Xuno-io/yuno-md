from .neibot_service_interface import NeibotServiceInterface
from xuno_components.logger.logger_interface import LoggerInterface
from openai import OpenAI
from openai.types.chat import ChatCompletion
import logging

class NeibotService(NeibotServiceInterface):
    def __init__(self, openai_client: OpenAI, logger: logging.Logger) -> None:
        self.openai_client: OpenAI = openai_client
        self.logger: logging.Logger = logger
        
    def get_response(self, history: list[tuple[str, str]]) -> str:
        # first element is the role and second is the message
        messages = [{"role": role, "content": content} for role, content in history]
        
        try:
            response: ChatCompletion = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150,
                n=1,
                stop=None,
                temperature=0.7,
            )
            message_content = response.choices[0].message.content
            if isinstance(message_content, str):
                return message_content.strip()

            return "".join(part.get("text", "") for part in message_content).strip()
        except Exception as e:
            self.logger.error(f"Error getting response from OpenAI: {e}")
            return "I'm sorry, I couldn't process your request at the moment."
