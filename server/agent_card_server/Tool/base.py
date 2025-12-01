from pydantic import BaseModel 

class BaseTool(BaseModel): 
    """"""
    class Config: 
        arbitrary_types_allowed = True