from chatflare.agent.base import BaseAgent
from chatflare.graph.workflow import BaseWorkflow
from agent_card_server.Memory.base import MemoryBank
from chatflare.tracker.base import Blob, Commit, Branch
from chatflare.tracker.utils import visualize_branches
from chatflare.graph.base import GraphState, GraphTraverseThread
from chatflare.agent.task import SRTask
from threading import Thread


from chatflare.graph.nodes import LLMTaskNode, HumanInstructionNode, ENDNODE
from chatflare.graph.action import BaseAction

from agent_card_server.ActionChatFlare.retrieve import RetrieveAction
from agent_card_server.ActionChatFlare.read import ReadAction
from agent_card_server.ActionChatFlare.synthesis import SynthesizeAction 
from agent_card_server.ActionChatFlare.discuss import DiscussAction

from typing import Any, Callable, Dict, List, Optional, Tuple

import asyncio

class IRAgent(BaseAgent):
    def __init__(self, agent_id: str=None, agent_name: str=None, workflow:BaseWorkflow=None, environment=None, thread=None, task_at_hand=None, client_handler=None, TRAVERSE_MAX_DEPTH=5): 
        super().__init__(agent_id, agent_name, workflow, environment, thread)
        self.environment = environment
        self.workflow:BaseWorkflow = workflow 
        self.task_at_hand = task_at_hand
        self.client_handler = client_handler
        self.workflow.TRAVERSE_MAX_DEPTH = TRAVERSE_MAX_DEPTH
        self.current_task = None 
        self.loop = asyncio.new_event_loop() 
        self._thread = Thread(target=self._start_event_loop, daemon=True)
        self._thread.start()
        print("initialize agents")
        print(self.active_thread.graph_state.paper_visited_in_commits)

    
    def _start_event_loop(self): 
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        
    def assign_environment(self, environment):
        self.environment = environment
        if self not in environment.agents:
            environment.agents.append(self)
        self.active_thread.graph_state.environment = environment
        if environment.task is not None:
            self.task_at_hand.research_query = environment.task

            
    def assign_task(self, task, environment: Any = None, client_handler: Any = None, **kwargs):
        """Assign a task to the agent"""
        self.task_at_hand.research_query = task
        if environment is not None:
            self.assign_environment(environment)
        self.client_handler = client_handler
        task_res = self.astart_workflow() 
        return task_res
    
    def stop_event_loop(self): 
        if self.loop.is_running():
            self.liio.call_soon_threadsafe(self.loop.stop)
        self.thread.join() 
   
    def stop_workflow(self): 
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()

    def astart_workflow(self): 
        # try: 
        #     asyncio.create_task(self.workflow.arun())
        # except RuntimeError as e:
        self.stop_workflow()
        self.current_task = asyncio.run_coroutine_threadsafe(self.workflow.arun(), self.loop) 
            
    # def astart_workflow_deprecated(self): 
    #     try:
    #         # in jupyter env
    #         # Use the current running loop instead of creating a new one.
    #         asyncio.create_task(self.workflow.arun())
    #         # self.async_start_new_task()
    #     except RuntimeError as e:
    #         print("Try Normal Env")
    #         # in normal python env
    #         loop = asyncio.new_event_loop()
    #         asyncio.set_event_loop(loop)
    #         try: 
    #             # loop.run_until_complete(asyncio.create_task(self.async_start_new_task()))
    #             loop.run_until_complete(self.workflow.arun())
    #         finally:
    #             loop.close()

    def human_instruction_on_paper(self, fromPaperId, toPaperId, client_handler=None):
        if not self.client_handler: 
            self.client_handler = client_handler
        
        ## check self.active_thread.branch.commits.blobs[0].result["meta"]["paper_id"], if it is the same as fromPaperId, then thread.rollback(commitId)
        target_commit = None 
        for commit in self.active_thread.branch.commits:
            if commit.blobs[0].result["meta"].get("paper_id") and commit.blobs[0].result["meta"].get("paper_id") == fromPaperId:
                target_commit = commit
                break
        
        if target_commit is not None:
            print("find target commit!!!")
            print("\n")
            new_thread = self.active_thread.rollback(target_commit)
            new_thread.graph_state.agent = self
            new_thread.graph_state.environment = self.environment
            
            self.active_thread = new_thread
            self.workflow.traverse_thread = new_thread
            self.threads[new_thread.thread_id] = new_thread
            toPaperObject = self.active_thread.graph_state.environment.literature_bank.docstore._dict[toPaperId] 
            
            self.active_thread.graph_state.cached_work["papers_to_read"] = [toPaperObject]
            if self.workflow.working_status != "RUNNING":
                self.workflow.current_node = self.workflow.read_node
                self.resume_workflow()
            else:
                if len(self.active_thread.graph_state.cached_work["papers_to_synthesize"]) != 0: 
                    self.active_thread.graph_state.cached_work["papers_to_synthesize"] = []
                     
            print("new thread created!!!")
            print("\n")
            print("new thread created!!!")
            print("\n")
            print("new thread created!!!")
            print("\n")
        else:
            fromPaperObject = self.active_thread.graph_state.environment.literature_bank.docstore._dict[fromPaperId]
            toPaperObject = self.active_thread.graph_state.environment.literature_bank.docstore._dict[toPaperId] 
            self.active_thread.graph_state.cached_work["papers_to_read"] = [toPaperObject, fromPaperObject] 
            if self.workflow.working_status != "RUNNING":
                self.workflow.current_node = self.workflow.read_node
                self.resume_workflow()
            else:
                if len(self.active_thread.graph_state.cached_work["papers_to_synthesize"]) != 0: 
                    self.active_thread.graph_state.cached_work["papers_to_synthesize"] = []
        
        ## else add two papers to thread.graph_state.paper_to_read with document objec

    
    def pause_workflow(self):
        self.workflow.pause_traverse()

    def resume_workflow(self):
        try:
            # in jupyter env
            # Use the current running loop instead of creating a new one.
            asyncio.create_task(self.workflow.resume_traverse())
            # self.async_start_new_task()
        except RuntimeError as e:
            print("Try Normal Env")
            # in normal python env
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try: 
                # loop.run_until_complete(asyncio.create_task(self.async_start_new_task()))
                loop.run_until_complete(self.workflow.resume_traverse())
            finally:
                loop.close()

    def communicate_with_agent(self, command): 
        try:
            # in jupyter env
            # Use the current running loop instead of creating a new one.
            asyncio.create_task(self.workflow.take_human_instruction(command))
            # self.async_start_new_task()
        except RuntimeError as e:
            print("Try Normal Env")
            # in normal python env
            # loop = asyncio.new_event_loop()
            # asyncio.set_event_loop(loop)
            # try: 
            #     # loop.run_until_complete(asyncio.create_task(self.async_start_new_task()))
            #     loop.run_until_complete(self.workflow.take_human_instruction(command))
            # finally:
            #     loop.close()
            if not self.current_task or self.current_task.done():
                self.current_task = asyncio.run_coroutine_threadsafe(self.workflow.take_human_instruction(command, True), self.loop)
            else:
                self.current_task = asyncio.run_coroutine_threadsafe(self.workflow.take_human_instruction(command), self.loop)

                

def to_retrieve_condition(thread): 
    return not thread.graph_state.finish_all_articles \
        and len(thread.graph_state.human_cached_instructions) == 0\
        and len(thread.graph_state.cached_work['papers_to_read']) == 0\
        and len(thread.graph_state.human_cached_instructions) == 0
# and len(thread.graph_state.cached_work['papers_to_synthesize']) == 0 \

def to_read_condition(thread):
    return  len(thread.graph_state.human_cached_instructions) == 0\
        and len(thread.graph_state.cached_work['papers_to_read']) > 0 \
        and (len(thread.graph_state.cached_work['papers_to_synthesize']) == 0 or (len(thread.graph_state.cached_work['papers_to_synthesize']) > 0 and len(thread.graph_state.memory.docstore._dict) < 2)) \
        and len(thread.graph_state.human_cached_instructions) == 0
        
def to_synthesis_condition(thread):
    return len(thread.graph_state.human_cached_instructions) == 0\
        and len(thread.graph_state.cached_work['papers_to_synthesize']) > 0 \
        and len(thread.graph_state.human_cached_instructions) == 0 \
        and len(thread.graph_state.memory.docstore._dict) > 1 
        
def to_end_condition(thread):
    return thread.graph_state.finish_all_articles \
        and len(thread.graph_state.cached_work['papers_to_read']) == 0 
        # and len(thread.graph_state.cached_work['papers_to_synthesize']) == 0 \

def to_discuss_condition(thread):
    return len(thread.graph_state.human_cached_instructions) > 0
               

def create_ir_agent(agent_id, TRAVERSE_MAX_DEPTH=20) -> IRAgent: 
    taskObj = SRTask(research_query=None)
    memory = MemoryBank.initialize_memory()
    graph_state = GraphState(environment=None, memory=memory)
    thread = GraphTraverseThread(graph_state=graph_state, task=taskObj)

    
    retrieveAction = RetrieveAction(action_name="retrieve")
    retrieveNode = LLMTaskNode("retrieve", retrieveAction)
    
    readAction = ReadAction(action_name="read")
    readNode = LLMTaskNode("read", readAction)
    
    synthesisAction = SynthesizeAction(action_name="synthsis")
    synthesisNode = LLMTaskNode("synthesis", synthesisAction)                

    discussAction = DiscussAction(action_name="discuss")
    discussNode = HumanInstructionNode("discuss", discussAction)

    endnode = ENDNODE()

    agent_workflow = BaseWorkflow(thread)

    agent_workflow.add_node(retrieveNode)
    agent_workflow.add_node(readNode)
    agent_workflow.add_node(synthesisNode)
    agent_workflow.add_node(discussNode)
    agent_workflow.add_node(endnode)

    agent_workflow.add_edge(retrieveNode, readNode, to_read_condition)
    agent_workflow.add_edge(retrieveNode, retrieveNode, to_retrieve_condition)
    agent_workflow.add_edge(readNode, synthesisNode, to_synthesis_condition)
    agent_workflow.add_edge(readNode, retrieveNode, to_retrieve_condition)
    agent_workflow.add_edge(readNode, readNode, to_read_condition)
    agent_workflow.add_edge(synthesisNode, readNode, to_read_condition)
    agent_workflow.add_edge(synthesisNode, retrieveNode, to_retrieve_condition)
    agent_workflow.add_edge(retrieveNode, discussNode, to_discuss_condition)
    agent_workflow.add_edge(readNode, discussNode, to_discuss_condition)
    agent_workflow.add_edge(synthesisNode, discussNode, to_discuss_condition)
    agent_workflow.add_edge(discussNode, retrieveNode, to_retrieve_condition)
    agent_workflow.add_edge(discussNode, readNode, to_read_condition)
    agent_workflow.add_edge(discussNode, synthesisNode, to_synthesis_condition)
    agent_workflow.add_edge(retrieveNode, endnode, to_end_condition)

    agent_workflow.start_node = retrieveNode
    agent_workflow.read_node = readNode 
    agent_workflow.retrieve_node = retrieveNode
    agent_workflow.synthesis_node = synthesisNode
    agent_workflow.set_end_node(endnode)
    
    agent = IRAgent(agent_id=agent_id, agent_name="IRAgent", workflow=agent_workflow, environment=None, thread=thread, task_at_hand=taskObj, client_handler=None, TRAVERSE_MAX_DEPTH=TRAVERSE_MAX_DEPTH)
    graph_state.agent = agent 
    thread.agent = agent 
    return agent 
    