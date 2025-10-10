from .neibot_service_interface import NeibotServiceInterface
from xuno_components.logger.logger_interface import LoggerInterface
from openai import OpenAI
from openai.types.chat import ChatCompletion
import logging

class NeibotService(NeibotServiceInterface):
    def __init__(self, system_prompt: str, model_name: str , openai_client: OpenAI, logger: logging.Logger) -> None:
        self.openai_client: OpenAI = openai_client
        self.logger: logging.Logger = logger
        self.model_name: str = model_name
        self.system_prompt: str = system_prompt
        
        self.logger.info(f"NeibotService initialized with model: {self.model_name}")
        self.logger.info(f"System prompt: {self.system_prompt}")
        
    def get_response(self, history: list[tuple[str, str]]) -> str:
        # first element is the role and second is the message
        history.insert(0, ("system", self.system_prompt))
        messages = [{"role": role, "content": content} for role, content in history]
        
        try:
            response: ChatCompletion = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
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
