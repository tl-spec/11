from Agent.base import BaseAgent
from Action.base import BaseAction
from Action.retrieve import RetrieveAction, Read
from langchain.docstore.document import Document
from typing import List, Dict, Optional, Union, Tuple
from Memory.base import MemoryBank

class IRAgent(BaseAgent):
    """Agent that perform Information Seeking"""
    agent_id: str = "IR_agent"
    memory: MemoryBank = MemoryBank.initialize_memory()
    actions: List[BaseAction] = [RetrieveAction(), Read()]
    action_sequence_done: List[BaseAction] = []
    workspace: List[Document] = []
    agent_working_memory: str = ""
    agent_scratchpad: str = ""
    task_at_hand: str = ""
    max_work_iterations: int = 5
    current_iter: int = 0
    _iter: int = 0 
    observation: Tuple[str, Dict] = None
    papers_to_read: List[Document] = []

    def assign_task(self, task: str, workload: List[Document]):
        """Assign a task to the agent"""
        self.task_at_hand = task
        self.workspace = workload
        task_res = self.start_task() 
        return task_res

    def initialize_task_status(self):
        self.current_iter = 0 
        self.observation = None 
        self.papers_to_read = []     
        

    def start_task(self):
        """start the task"""
        observations = [] 
        self.initialize_task_status() 
        while self.current_iter < self.max_work_iterations:
            observations.append(self.act())
        return observations


    def act(self, query: str = None):
        """"""
        if query is None:
            query = self.task_at_hand
        next_available_action = self.get_next_action_planning(query)
        observation = self.take_action(next_available_action)
        self.current_iter += 1
        return observation

    def take_action(self, action: BaseAction, query: str = None):
        """Take an action"""
        if query is None:
            query = self.task_at_hand
        if action.action_name == "retrieve":
            self.observation = action.act( ## if this is retrieve action, it will probably add article to to read list
                agent=self,
                query=query, 
                scratch_pad=self.agent_scratchpad,
                working_memory=self.agent_working_memory, 
                memory=self.memory, 
                workspace=self.workspace
            )
            if self.observation[0] == "no more paper to read": 
                self.current_iter = self.max_work_iterations
                return self.observation
            self.papers_to_read = self.observation[1]
        elif action.action_name == "read":
            """"""
            self.observation = action.act(
                agent=self,
                query=query, 
                scratch_pad=self.agent_scratchpad,
                working_memory=self.agent_working_memory, 
                memory=self.memory, 
                workspace=self.papers_to_read.pop()
            )
        return self.observation

    def get_next_action_planning(self, input=None):
        """
        Depends on whether the iteration has finished, planning next move
        """
        if self.current_iter >= self.max_work_iterations: return 

        if len(self.papers_to_read) == 0:         
            return self.actions[0] # retrieve action
        else:
            return self.actions[1] # read action
        

        # if self.current_iter < self.max_work_iterations:
        #     return self.actions[self._iter]

       
