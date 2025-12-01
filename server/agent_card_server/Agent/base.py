# 导入pydantic的BaseModel,用于数据模型定义和验证
from pydantic import BaseModel
from typing import List, Any
from agent_card_server.Action.base import BaseAction
from agent_card_server.Memory.base import MemoryBank
# from agent_card_server.connection.connection_handler import ClientConnectionHandler
# from Environment.environment import BaseEnvironment

#定义BaseAgent基类，继承自BaseModel（pydantic的模型类，提供数据验证等功能）
class BaseAgent(BaseModel): 
    """"""
    #智能体的唯一标识ID
    agent_id: str
    #私有属性_memory，类型为MemoryBank,初始值为None
    _memory: MemoryBank = None
    #智能体可执行的动作列表，元素类型为BaseAction
    actions: List[BaseAction] = []
    #当前所处的环境对象，类型暂定为object
    environment_cur: object = None# environment at hand
    #智能体的角色，默认为”agent"
    role: str = "agent"
    #管理员命令列表，存储接收的管理指令
    supervisor_commands: List[str] = []
    #客户端处理器，类型为Any(可接受任意类型，后续可细化)
    client_handler: Any = None

    #配置类，用于pydantic模型的额外设置
    class Config:
        #允许模型包含任意类型的属性（突破pydantic默认的类型限制）
        arbitrary_types_allowed = True
    @property#定义memory属性的getter方法
    def memory(self):
        #如果memory未初始化（为None），则调用MemoryBank的initialize_memory方法初始化
        if not self._memory:
            self._memory = MemoryBank.initialize_memory()
        #返回初始化后的_memory
        return self._memory

#定义HumanProxy类，继承自BaseAgent基类（人类代理，可能用于人类对智能体的管理）
class HumanProxy(BaseAgent):
    """"""
    #重写BaseAgent的role属性，默认为”supervisor"
    role: str = "supervisor"
    #配置类，与BaseAgent保持一致，允许任意类型属性
    class Config: 
        arbitrary_types_allowed = True
    #定义speak方法，用于人类代理向智能体列表发送动作指令
    def speak(self, agents: List[BaseAgent], action: str):
        """"""
        #抛出未实现错误，提示该方法需要在子类中实现
        raise NotImplementedError
    #定义broadcase方法：人类代理向智能体列表广播信息
    def broadcase(self, agents):
        """"""
        # 抛出未实现错误，提示该方法需要在子类中实现
        raise NotImplementedError


