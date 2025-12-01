from pydantic import BaseModel
from agent_card_server.Tool.base import BaseTool 
from typing import List, Dict, Optional, Union
from langchain.docstore.document import Document
# from Agent.base import BaseAgent
from abc import ABC, abstractmethod
from agent_card_server.Memory.base import MemoryBank

class BaseAction(BaseModel):
    action_name: str
    prompt_prefix: str
    allowed_tools: List[BaseTool] = []

    class Config: 
        arbitrary_types_allowed = True 

    @abstractmethod
    def act(self, agent, query: str, scratch_pad: str, working_memory: str, memory: MemoryBank, workspace: Union[List[Document], Document]):
        """act"""