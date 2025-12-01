#导入必要的类和模块
from agent_card_server.Agent.base import BaseAgent #导入基础智能体类
from agent_card_server.Action.base import BaseAction #导入基础动作类
from agent_card_server.Action.retrieve import RetrieveAction, Read, Discuss #导入检索、阅读、讨论动作类
from agent_card_server.Action.summarize import SummarizeAction #导入总结动作类
from agent_card_server.Action.overall_report import OverallSummarizeAction #导入整体报告动作类
from langchain.docstore.document import Document #导入文档类（用于处理文献）
from typing import List, Dict, Optional, Union, Tuple, Any #导入类型提示工具
from agent_card_server.Memory.base import MemoryBank #导入记忆库类
from agent_card_server.Environment.environment import BaseEnvironment, InformationSeekingEnvironment #导入环境类
from agent_card_server.Output.print import log_pause, log_errors #导入日志打印工具
import asyncio #导入异步处理模块

#定义信息检索智能体类，继承自基础智能体类
class IRAgent(BaseAgent):
    """Agent that perform Information Seeking"""#文档字符串：执行信息检索的代理
    agent_id: str = "IR_agent"#代理唯一标识
    #定义可用动作：检索、阅读、讨论、总结、整体报告
    procedures: Dict[str, BaseAction] = {"retrieve": RetrieveAction(), "read": Read(), "discuss": Discuss(), "summarize": SummarizeAction(), "overall_report": OverallSummarizeAction()}
    # action_sequence_done: List[Dict[str, str]] = []
    #当前工作空间（存储环境相关数据）
    workspace: Any = [] #current environment
    environment_cur: BaseEnvironment = None    #当前环境实例
    """agent's working memory consists of observation and current working memeory string"""
    agent_working_memory: str = "" #代理的工作记忆字符串
    observation: Tuple[str, Dict] = None #当前观察结果（元组：状态+数据）
    observations: List[Tuple[str, Dict]]= []#所有观察结果的列表
    action_sequence_done: List[Dict] = [] #已完成的动作序列记录
    #代理与监督者的对话历史
    agent_scratchpad: List[Dict[str, str]] = [] # contains agent's conversation history with supervisors.
    inspiration_conversation_history: str = "" #启发式对话历史
    report_instructions: Optional[str] = None #报告生成指令
    task_at_hand: str = ""#当前任务描述

    """workload parameters"""#工作负载参数
    max_work_iterations: int = 50#最大工作迭代次数
    current_iter: int = 0 #当前迭代次数
    _iter: int = 0  # 内部迭代计数器
    papers_to_read: List[Document] = [] #待阅读的文献列表
    papers_visited: List[Document] = [] #已访问的文献列表
    supervisor_commands: List[str] = [] #监督者的指令列表
    working_status: str = "idle" ## 工作状态：idle（空闲） | retrieve（检索） | read（阅读） | discuss（讨论）
    #为智能体分配环境
    def assign_environment(self, environment: BaseEnvironment):
        self.environment_cur = environment  #设置当前环境
        if self not in environment.agents:  #如果智能体不在环境的智能体列表中
            environment.agents.append(self)  #将智能体添加到环境
    #为智能体分配任务
    def assign_task(self, task, environment: Any = None, workspace: Any = None, client_handler: Any = None):
        """Assign a task to the agent"""#文档字符串：为智能体分配任务
        self.task_at_hand = task #设置当前任务
        self.environment_cur = environment or self.environment_cur #设置当前环境
        self.workspace = workspace or environment.literature_bank.docstore._dict.values() #设置工作空间
        self.client_handler = client_handler #设置客户端处理器
        task_res = self.astart_task()  #异步启动任务
        return task_res #返回任务结果

 ##初始化任务状态
    def initialize_task_status(self):
        self.current_iter = 0 #重置当前迭代次数
        self.observation = None  #清空当前观察结果
        self.observations = [] #清空观察结果列表
        self.papers_to_read = []   #清空待阅读的文献列表
        self.working_status = "retrieve"  #设置工作状态为检索
   #异步启动新任务
    async def async_start_new_task(self): 
        """start the task"""#文档字符串：启动任务
        self.initialize_task_status() #初始化任务状态
        #循环条件：未达到最大迭代次数，或有待阅读文献，或记忆中有最新总结
        while self.current_iter < self.max_work_iterations or len(self.papers_to_read) > 0 or self.memory.latest_summary is not None:
            if self.working_status == "discuss": #工作状态为讨论
                self.observations.append(await self.aact())  #执行异步动作并添加观察结果
                break
            elif self.working_status == "idle": #工作状态为空闲
                log_pause() #暂停
                break
            else: 
                self.observations.append(await self.aact()) #执行异步动作并记录观察结果
        if self.current_iter >= self.max_work_iterations: #若达到最大迭代次数
            self.working_status = "idle" #设置工作状态为空闲
        return self.observations #返回观察结果列表
    #异步恢复任务
    async def async_resume_new_task(self): 
        """Resume the task"""#文档字符串：恢复任务
        #循环条件：未达到最大迭代次数，或有待阅读文献
        while self.current_iter < self.max_work_iterations or len(self.papers_to_read) > 0:
            if self.working_status == "discuss": #工作状态为讨论
                self.observations.append(await self.aact()) #执行异步动作并添加观察结果
                break
            elif self.working_status == "idle":#工作状态为空闲
                break
            else: 
                self.observations.append(await self.aact())#执行异步动作并记录观察结果
        if self.current_iter >= self.max_work_iterations:#若达到最大迭代次数
            self.working_status = "idle"
        return self.observations
    #异步启动任务的入口方法
    def astart_task(self): 
        # asyncio.create_task(self.async_start_new_task())
        try:
            # in jupyter env在jupyter环境中，使用当前事件循环创建任务
            # Use the current running loop instead of creating a new one.
            asyncio.create_task(self.async_start_new_task())
            # self.async_start_new_task()
        except RuntimeError as e:#捕获运行时错误（如无当前循环）
            print("Try Normal Env")#提示切换到普通环境
            # in normal python env在普通Python环境中，创建新事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try: 
                # loop.run_until_complete(asyncio.create_task(self.async_start_new_task()))
                loop.run_until_complete(self.async_start_new_task())#运行异步任务，直到完成
            finally:
                loop.close()#关闭事件循环
    #启动整体报告生成
    def astart_overall_report(self, instructions: Optional[str] = None): 
        self.report_instructions = instructions#设置报告指令
        try: 
            asyncio.create_task(self.async_start_overall_report())#异步创建报告任务
        except RuntimeError as e:#捕获运行时错误
            loop = asyncio.new_event_loop()#创建新事件循环
            asyncio.set_event_loop(loop)
            try: 
                loop.run_until_complete(self.async_start_overall_report())#运行报告任务
            finally:
                loop.close()#关闭循环
    #异步生成整体报告
    async def async_start_overall_report(self):
        action = self.procedures["overall_report"] #获取整体报告动作
        query = self.task_at_hand #任务作为查询
        #执行整体报告动作
        self.observation = await action.act(
                agent=self,
                query=query, 
                scratch_pad=self.agent_scratchpad,
                working_memory=self.agent_working_memory, 
                memory=self.memory, 
        )
    #恢复任务的入口方法
    def aresume_task(self):
        if self.papers_to_read: #待阅读文献列表不为空
            self.working_status = "read" #设置工作状态为阅读
        else:
            self.working_status = "retrieve"#设置工作状态为检索
        try:
            asyncio.create_task(self.async_resume_new_task())#异步恢复任务
        except RuntimeError as e:#捕获运行时错误
            loop = asyncio.new_event_loop()#创建新事件循环
            asyncio.set_event_loop(loop)#将该循环设置为当前线程的事件循环，用于执行异步任务
            try: 
                loop.run_until_complete(self.async_resume_new_task())#运行恢复任务直到协程完成
            finally:
                loop.close()#关闭事件循环
#同步启动任务(已被异步方法替代，可用于兼容)
    def start_task(self):
        """start the task"""#文档字符串：启动任务
        observations = [] #观察结果列表
        self.initialize_task_status() #初始化任务状态
        while self.current_iter < self.max_work_iterations:#循环条件：未达到最大迭代次数
            observations.append(self.act())#执行动作并添加观察结果
        return observations
    #同步执行动作
    def act(self, query: str = None):
        """"""#空文档字符串
        if query is None:#若未提供查询
            query = self.task_at_hand  #使用当前任务作为查询
        next_available_action = self.get_next_action_planning(query) #获取下一个可用动作
        observation = self.take_action(next_available_action) #执行动作并记录观察信息
        if next_available_action.action_name != "discuss": self.current_iter += 1 #如果动作不是讨论，则迭代次数加1
        return observation

    async def aact(self, query: str = None):
        """"""
        if query is None:#未提供查询
            query = self.task_at_hand #使用当前任务作为查询
        next_available_action = self.get_next_action_planning(query) #获取下一个可用动作
        if next_available_action is None: #无可用动作
            log_errors("No more action to take")#错误日志：无更多动作可执行
            return
        observation = await self.atake_action(next_available_action)#异步执行动作
        if next_available_action.action_name != "discuss" and next_available_action.action_name != "summarize": self.current_iter += 1
        #如果动作不是讨论，则迭代次数加1
        return observation
    #同步执行具体行动
    def take_action(self, action: BaseAction, query: str = None):
        """Take an action"""
        if query is None:#未提供查询
            query = self.task_at_hand #使用当前任务作为查询

        if action.action_name == "discuss": #若动作是讨论
            command = self.supervisor_commands.pop()#取出最新监督指令
            #执行讨论动作
            self.observation = action.act(
                agent=self,
                query=query, 
                scratch_pad=self.agent_scratchpad,
                working_memory=self.agent_working_memory, 
                memory=self.memory, 
                workspace=self.workspace, 
                command=command
            )
            self.current_iter += 1
        elif action.action_name == "retrieve":#若动作是检索
            self.observation = action.act( ## 如果这是检索操作，它可能会将文章添加到阅读列表中
                agent=self,
                query=query, 
                scratch_pad=self.agent_scratchpad,
                working_memory=self.agent_working_memory, 
                memory=self.memory, 
                workspace=self.workspace
            )
            if self.observation[0] == "no more paper to read": #如果没有更多文章可读
                self.current_iter = self.max_work_iterations #设置迭代次数为最大值
                return self.observation
            self.papers_to_read = self.observation[1]#设置待阅读文献列表
        elif action.action_name == "read":#若动作是阅读
            """"""
            self.observation = action.act(
                agent=self,
                query=query, 
                scratch_pad=self.agent_scratchpad,
                working_memory=self.agent_working_memory, 
                memory=self.memory, 
                workspace=self.papers_to_read.pop()
            )
        self.action_sequence_done.append({"action": action.action_name, "observation": self.observation})#添加已完成的动作和观察信息
        return self.observation
#异步执行具体行动
    async def atake_action(self, action: BaseAction, query: str = None):
        """Take an action"""
        if query is None:#未提供查询
            query = self.task_at_hand#使用当前任务作为查询

        if action.action_name == "discuss": #若动作是讨论
            command = self.supervisor_commands.pop()#取出最新监督指令
            self.observation = await action.act(#异步执行讨论动作
                agent=self,
                query=query, 
                scratch_pad=self.agent_scratchpad,
                working_memory=self.agent_working_memory, 
                memory=self.memory, 
                workspace=self.workspace, 
                command=command
            )
        elif action.action_name == "retrieve":#若动作是检索
            self.observation = await action.act( ## 如果是检索动作，会将文章加入阅读列表
                agent=self,
                query=query, 
                scratch_pad=self.agent_scratchpad,
                working_memory=self.agent_working_memory, 
                memory=self.memory, 
                workspace=self.workspace
            )
            if self.observation[0] == "no more paper to read": #如果没有更多文章可读
                self.current_iter = self.max_work_iterations#设置迭代次数为最大值
                return self.observation#返回观察信息
            self.papers_to_read = self.observation[1]#设置待阅读文献列表
        elif action.action_name == "read":#若动作是阅读
            """"""
            self.observation = await action.act(#异步执行阅读动作
                agent=self,
                query=query, 
                scratch_pad=self.agent_scratchpad,
                working_memory=self.agent_working_memory, 
                memory=self.memory, 
                workspace=self.papers_to_read.pop()
            )

        elif action.action_name == "summarize":#若动作是总结
            self.observation = await action.act(#异步执行总结动作
                agent=self,
                query=query, 
                scratch_pad=self.agent_scratchpad,
                working_memory=self.agent_working_memory, 
                memory=self.memory, 
            )
        self.action_sequence_done.append({"action": action.action_name, "observation": self.observation})#添加已完成的动作和观察信息
        return self.observation
    #规划下一步动作
    def get_next_action_planning(self, input=None):
        """
        Depends on whether the iteration has finished, planning next move
        Step 1: check whether there is unresolved supervisor command
        """#取决于迭代是否完成，规划下一步步骤1：检查是否有未解析的supervisor命令
        commands_exist = self.check_superviser_command()#检查是否存在未解析的supervisor命令
        if commands_exist:#如果存在未解析的supervisor命令
            return self.procedures["discuss"]#返回讨论动作
        else:    #如果不存在未解析的supervisor命令
            #如果迭代次数已经达到最大值且没有待阅读文献且没有最新总结，则返回
            if self.current_iter >= self.max_work_iterations and len(self.papers_to_read) == 0 and self.memory.latest_summary is None: return
            # 如果上一次动作是阅读且存在最新总结，则返回总结动作
            if len(self.action_sequence_done) > 0 and self.action_sequence_done[-1]['action'] == "read" and self.memory.latest_summary is not None:
                return self.procedures["summarize"]#返回总结动作
            if len(self.papers_to_read) == 0:         #如果没有待阅读文献
                return self.procedures["retrieve"] ## retrieve action
            else:
                return self.procedures["read"] # read action
        
        # if self.current_iter < self.max_work_iterations:
        #     return self.actions[self._iter]
    #检查是否有监督指令
    def check_superviser_command(self) -> bool: 
        return len(self.supervisor_commands) > 0 #若指令列表非空则返回True
    #处理来自监督者的指令（异步）
    async def command_from_supervisor(self): 
        if self.working_status == "idle": #如果当前状态是空闲
            self.working_status = "discuss"#设置当前状态为讨论
            await self.aact()#执行异步动作
        elif self.working_status == "discuss":#如果当前状态是讨论
            await self.aact()#继续执行异步动作
        else:#其他状态
            self.working_status = "discuss"#切换为讨论状态
            ## wait for the current action to finish
    #沟通与智能体的指令（接收监督指令）
    def communicate_with_agent(self, command): 
        self.supervisor_commands.append(command)#添加指令到列表
        try:
            # in jupyter env在Jupyter环境中异步处理指令
            # Use the current running loop instead of creating a new one.
            asyncio.create_task(self.command_from_supervisor())
            # self.async_start_new_task()
        except RuntimeError as e:#运行时错误
            print("Try Normal Env")#提示切换环境
            # in normal python env在普通Python环境中处理指令
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try: 
                # loop.run_until_complete(asyncio.create_task(self.async_start_new_task()))
                loop.run_until_complete(self.command_from_supervisor())#执行指令处理
            finally:
                loop.close()#关闭循环
    #暂停任务
    def pause_task(self): 
        self.working_status = "idle"#设置当前状态为空闲
        
    #恢复任务
    def resume_task(self):
        self.working_status = "retrieve"#设置当前状态为检索
        self.act()#执行动作

