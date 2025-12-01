from chatflare.agent.base import BaseAgent
from chatflare.graph.workflow import BaseWorkflow
from agent_card_server.Memory.base import MemoryBank
from chatflare.tracker.base import Blob, Commit, Branch
from chatflare.tracker.utils import visualize_branches
from chatflare.graph.base import GraphState, GraphTraverseThread
from chatflare.agent.task import SRTask
from threading import Thread
from agent_card_server.connection.firebase_handler import update_agent_creation_in_study, update_agent_in_study


from chatflare.graph.nodes import LLMTaskNode, HumanInstructionNode, ENDNODE
from chatflare.graph.action import BaseAction

from agent_card_server.ActionChatFlare.retrieve import RetrieveAction
from agent_card_server.ActionChatFlare.read import ReadAction
from agent_card_server.ActionChatFlare.synthesis import SynthesizeAction 
from agent_card_server.ActionChatFlare.discuss import DiscussAction

from typing import Any, Callable, Dict, List, Optional, Tuple
import os
import asyncio

class IRAgent(BaseAgent):
    def __init__(self, agent_id: str=None, agent_name: str=None, workflow:BaseWorkflow=None, environment=None, thread=None, task_at_hand=None, agent_model=None, client_handler=None, TRAVERSE_MAX_DEPTH=5): 
        super().__init__(agent_id, agent_name, workflow, environment, thread)
        self.environment = environment
        self.workflow:BaseWorkflow = workflow 
        self.task_at_hand = task_at_hand
        self.client_handler = client_handler
        if self.workflow: 
            self.workflow.TRAVERSE_MAX_DEPTH = TRAVERSE_MAX_DEPTH
        self.current_task = None 
        self.loop = asyncio.new_event_loop() 
        self._thread = Thread(target=self._start_event_loop, daemon=True)
        self._thread.start()
        print("initialize agents")
        self.report_agent_status()
        self.report_agent_working_progress()
        self.agent_model = agent_model

        if self.client_handler and self.client_handler.in_user_study: 
            update_agent_creation_in_study(self.client_handler.user_study_config["user_study_id"], self.agent_id, self.agent_model, self.agent_name, self.task_at_hand.to_dict())

    def update_workflow_model(self, model_name, client_handler=None):    
        self.agent_model = model_name
        for node in self.workflow.nodes: 
            if node._NODE_TYPE == "LLM_TASK_NODE" or node._NODE_TYPE == "HUMAN_INSTRUCTION_NODE":
                node.action.switch_runnable_model(model_name)

        if client_handler is not None:
            self.client_handler = client_handler
        if self.client_handler:
            self.client_handler.emit("update_current_model", {"current_model": model_name, "agent_id": self.agent_id})
        
        if self.client_handler and self.client_handler.in_user_study: 
            update_agent_in_study(self.client_handler.user_study_config["user_study_id"], self.agent_id, self.agent_model, self.agent_name, self.task_at_hand.to_dict())

    
    def update_traverse_max_depth(self, TRAVERSE_MAX_DEPTH, client_handler=None):
        self.workflow.TRAVERSE_MAX_DEPTH = TRAVERSE_MAX_DEPTH
        if client_handler is not None:
            self.client_handler = client_handler
        if self.client_handler:
            self.report_agent_working_progress()
                
    def _start_event_loop(self): 
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def assign_environment(self, environment):
        self.environment = environment
        if self not in environment.agents:
            environment.agents.append(self)
        self.active_thread.graph_state.environment = environment
        if environment.task is not None and self.task_at_hand.research_query is None:
            self.update_research_question(environment.task)
        print("check assign environment")
        print(environment.task)



            
    def assign_task(self, task, environment: Any = None, client_handler: Any = None, **kwargs):
        """Assign a task to the agent"""
        if self.task_at_hand.research_query is None:
            self.update_research_question(task)
            
        if environment is not None:
            self.assign_environment(environment)
        
        if client_handler is not None:
            self.client_handler = client_handler
        task_res = self.astart_workflow() 
        return task_res

    def get_agent_working_status(self): 
        _status = {
            "total_document_visited": len(self.active_thread.graph_state.paper_visited_in_commits),    
            "total_document_included": len(self.active_thread.graph_state.memory.get_leaf_memories()),
        }   
        return _status

        
    def report_agent_working_progress(self, client_handler=None):
        if client_handler is not None:
            self.client_handler = client_handler
        if self.client_handler:
            self.client_handler.emit("update_review_progress", {"num_document_visited": len(self.active_thread.graph_state.visited_paperids), "num_document_included": len(self.active_thread.graph_state.memory.get_leaf_memories()), "agent_id": self.agent_id})
            self.client_handler.emit("update_agent_status", {"agent_id": self.agent_id, "status": self.workflow.working_status, "num_traversed": self.workflow.num_traversed, "traverse_max_depth": self.workflow.TRAVERSE_MAX_DEPTH, **self.get_agent_working_status()})
         
    def report_agent_status(self, client_handler=None, criteria_included=True):
        if client_handler is not None:
            self.client_handler = client_handler
        if self.client_handler:
            if self.environment:
                self.client_handler.emit("update_working_environment", {"working_environment": self.environment.clientCardId, "agent_id": self.agent_id})
            self.client_handler.emit("update_agent_status", {"agent_id": self.agent_id, "status": self.workflow.working_status, **self.get_agent_working_status()})
            self.client_handler.emit("update_research_question", {"research_question": self.task_at_hand.research_query, "agent_id": self.agent_id})
            self.client_handler.emit("update_user_specified_focus", {"user_specified_requirement": self.task_at_hand.user_specified_requirement, "agent_id": self.agent_id})
            self.client_handler.emit("update_inclusion_exclusion_criteria", {"inclusion_exclusion_criteria": self.task_at_hand.in_criteria, "agent_id": self.agent_id})
            self.client_handler.emit("update_summarization_requirement", {"summarization_requirement": self.task_at_hand.summarization_requirement, "agent_id": self.agent_id})  
    
    def update_research_question(self, research_question: str, client_handler=None):
        self.task_at_hand.research_query = research_question
        if client_handler is not None:
            self.client_handler = client_handler
        if self.client_handler:
            self.client_handler.emit("update_research_question", {"research_question": research_question, "agent_id": self.agent_id})
        
        if self.client_handler and self.client_handler.in_user_study: 
            update_agent_in_study(self.client_handler.user_study_config["user_study_id"], self.agent_id, self.agent_model, self.agent_name, self.task_at_hand.to_dict())

        self.add_interaction_commit("interaction-update-research-question", {
            "interaction-type": "interaction-update-research-question", 
            "interaction-detail": {
                "research_question": research_question
            }
        })


    def update_user_specified_requirement(self, user_specified_requirement: str, client_handler=None):
        self.task_at_hand.user_specified_requirement = user_specified_requirement
        if client_handler is not None:
            self.client_handler = client_handler
        if self.client_handler:
            self.client_handler.emit("update_user_specified_focus", {"user_specified_requirement": user_specified_requirement, "agent_id": self.agent_id})
        
        if self.client_handler and self.client_handler.in_user_study: 
            update_agent_in_study(self.client_handler.user_study_config["user_study_id"], self.agent_id, self.agent_model, self.agent_name, self.task_at_hand.to_dict())
        
        self.add_interaction_commit("interaction-update-user-specified-requirement", {
            "interaction-type": "interaction-update-user-specified-requirement", 
            "interaction-detail": {
                "user_specified_requirement": user_specified_requirement
            }
        })


    def update_inclusion_exclusion_critera(self, inclusion_exclusion_criteria: str, client_handler=None):
        self.task_at_hand.in_criteria = inclusion_exclusion_criteria
        if client_handler is not None:
            self.client_handler = client_handler
        if self.client_handler:
            self.client_handler.emit("update_inclusion_exclusion_criteria", {"inclusion_exclusion_criteria": inclusion_exclusion_criteria, "agent_id": self.agent_id})
        
        if self.client_handler and self.client_handler.in_user_study: 
            update_agent_in_study(self.client_handler.user_study_config["user_study_id"], self.agent_id, self.agent_model, self.agent_name, self.task_at_hand.to_dict())
        
        self.add_interaction_commit("interaction-update-inclusion-exclusion-criteria", {
            "interaction-type": "interaction-update-inclusion-exclusion-criteria", 
            "interaction-detail": {
                "inclusion_exclusion_criteria": inclusion_exclusion_criteria
            }
        })


    def update_summarization_requirement(self, summarization_requirement: str, client_handler=None):
        self.task_at_hand.summarization_requirement = summarization_requirement
        if client_handler is not None:
            self.client_handler = client_handler
        if self.client_handler:
            self.client_handler.emit("update_summarization_requirement", {"summarization_requirement": summarization_requirement, "agent_id": self.agent_id})
        
        if self.client_handler and self.client_handler.in_user_study: 
            update_agent_in_study(self.client_handler.user_study_config["user_study_id"], self.agent_id, self.agent_model, self.agent_name, self.task_at_hand.to_dict())

        self.add_interaction_commit("interaction-update-summarization-requirement", {
            "interaction-type": "interaction-update-summarization-requirement", 
            "interaction-detail": {
                "summarization_requirement": summarization_requirement
            }
        })

        
    def stop_event_loop(self): 
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join() 
   
    def stop_workflow(self): 
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
        # self.workflow.working_status == "PAUSED"

    def astart_workflow(self): 
        self.stop_workflow()
        ## Handle logic that agent just finished chat with human, current_node is discuss_node, then just route 
        next_node=None
        if self.workflow.working_status == "PAUSED" and self.workflow.current_node._NODE_TYPE == "HUMAN_INSTRUCTION_NODE": 
            next_node = self.workflow.route(self.workflow.current_node)
        self.current_task = asyncio.run_coroutine_threadsafe(self.workflow.arun(next_node), self.loop) 

    def areport(self, agents, cardId, client_handler, query=None, inclusion_exclusion_criteria=None, user_specified_requirement=None, summarization_requirement=None):
        self.stop_workflow()
        asyncio.run_coroutine_threadsafe(self.workflow.report_generation(agents, cardId, client_handler, query=query, inclusion_exclusion_criteria=inclusion_exclusion_criteria, user_specified_requirement=user_specified_requirement, summarization_requirement=summarization_requirement), self.loop)

    def human_instruction_on_paper(self, fromPaperId, toPaperId, client_handler=None):
        
        if not self.client_handler: 
            self.client_handler = client_handler
        
        ## check self.active_thread.branch.commits.blobs[0].result["meta"]["paper_id"], if it is the same as fromPaperId, then thread.rollback(commitId)
        target_commit = None 
        for commit in self.active_thread.branch.commits:
            if commit.blobs[0].result.get("meta") and commit.blobs[0].result["meta"].get("paper_id") and commit.blobs[0].result["meta"].get("paper_id") == fromPaperId:
                target_commit = commit
                break
        
        if target_commit is not None:
            new_thread = self.active_thread.rollback(target_commit)
            new_thread.graph_state.agent = self
            new_thread.agent = self
            new_thread.graph_state.environment = self.environment
            
            self.active_thread = new_thread
            self.workflow.traverse_thread = new_thread
            self.threads[new_thread.thread_id] = new_thread
            toPaperObject = self.active_thread.graph_state.environment.literature_bank.docstore._dict[toPaperId] 
            
            self.active_thread.graph_state.cached_work["papers_to_read"] = [toPaperObject]
            if self.workflow.working_status != "RUNNING":
                self.workflow.current_node = self.workflow.read_node
            else:
                if len(self.active_thread.graph_state.cached_work["papers_to_synthesize"]) != 0: 
                    self.active_thread.graph_state.cached_work["papers_to_synthesize"] = []
            
            self.add_interaction_commit("interaction-path", {
                "interaction-type": "interaction-path", 
                "interaction-detail": {
                    "from_paper_id": fromPaperId or 'none',
                    "to_paper_id": toPaperId or 'none'    
                }  
            })
            
            self.astart_workflow()
            self.report_agent_working_progress()
            self.report_agent_status()
        else:
            fromPaperObject = self.active_thread.graph_state.environment.literature_bank.docstore._dict.get(fromPaperId) if fromPaperId is not None else None
            toPaperObject = self.active_thread.graph_state.environment.literature_bank.docstore._dict.get(toPaperId) if toPaperId is not None else None
            to_read_list = [] 
            
            if toPaperObject:
                to_read_list.append(toPaperObject)
            if fromPaperObject:
                to_read_list.append(fromPaperObject)

            self.active_thread.graph_state.cached_work["papers_to_read"] = to_read_list
            if self.workflow.working_status != "RUNNING":
                self.workflow.current_node = self.workflow.read_node
            else:
                if len(self.active_thread.graph_state.cached_work["papers_to_synthesize"]) != 0: 
                    self.active_thread.graph_state.cached_work["papers_to_synthesize"] = []

            self.add_interaction_commit("interaction-path", {
                "interaction-type": "interaction-path", 
                "interaction-detail": {
                    "from_paper_id": fromPaperId or 'none',
                    "to_paper_id": toPaperId or 'none'    
                }  
            })
            self.astart_workflow()
        
        ## else add two papers to thread.graph_state.paper_to_read with document objec

    def pause_workflow(self):
        self.stop_workflow()
        if len(self.active_thread.branch.commits) > 0:
            self.active_thread.rollback(self.active_thread.branch.commits[-1], inplace=True)
        self.workflow.working_status = "PAUSED"
        self.report_agent_status()
        

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
        
        self.add_interaction_commit("interaction-communication", {
            "interaction-type": "interaction-communication", 
            "interaction-detail": {
                "command": command
            }
        })
        
        if not self.current_task or self.current_task.done():
            self.current_task = asyncio.run_coroutine_threadsafe(self.workflow.take_human_instruction(command), self.loop)
        else:
            self.stop_workflow()
            if len(self.active_thread.branch.commits) > 0:
                self.active_thread.rollback(self.active_thread.branch.commits[-1], inplace=True)
            print(self.active_thread.graph_state.cached_work)
            self.current_task = asyncio.run_coroutine_threadsafe(self.workflow.take_human_instruction(command), self.loop)
        
    def add_interaction_commit(self, task_type, blob):
        interaction_blob = Blob(task_type, blob)
        commit = Commit.from_blobs([interaction_blob], human_interaction=True)
        self.active_thread.add_commit(commit)

                

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
               
class FakeAgent: 
    def __init__(self, agent_id, client_handler):
        self.agent_id = agent_id
        self.client_handler = client_handler

def create_ir_agent(agent_id, client_handler, model_name=os.getenv("LLM_MODEL_NAME"), TRAVERSE_MAX_DEPTH=20) -> IRAgent: 
    ## check if research_query, in_criteria, user_specified_requirement, summarization_requirement is provided in os environment(loaded from .env), if not, use default value
    research_query = os.getenv("RESEARCH_QUERY") or None
    in_criteria = os.getenv("INCLUSION_EXCLUSION_CRITERIA") or None
    user_specified_requirement = os.getenv("USER_SPECIFIED_REQUIREMENT") or None
    summarization_requirement = os.getenv("DOMAIN_SPECIFIC_SUMMARY_PROMPT") or None
    taskObj = SRTask(research_query=research_query, in_criteria=in_criteria, user_specified_requirement=user_specified_requirement, summarization_requirement=summarization_requirement)
    memory = MemoryBank.initialize_memory()
    graph_state = GraphState(environment=None, memory=memory, agent=FakeAgent(agent_id=agent_id, client_handler=client_handler)) ## need to add a fake agent here to make sure graphstate can create a branch with agentid
    thread = GraphTraverseThread(graph_state=graph_state, task=taskObj, agent=FakeAgent(agent_id=agent_id, client_handler=client_handler))

    
    retrieveAction = RetrieveAction(action_name="retrieve", model_name=model_name)
    retrieveNode = LLMTaskNode("retrieve", retrieveAction)
    
    readAction = ReadAction(action_name="read", model_name=model_name)
    readNode = LLMTaskNode("read", readAction)
    
    synthesisAction = SynthesizeAction(action_name="synthsis", model_name=model_name)
    synthesisNode = LLMTaskNode("synthesis", synthesisAction)                

    discussAction = DiscussAction(action_name="discuss", model_name=model_name)
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
    
    agent = IRAgent(agent_id=agent_id, agent_name="IRAgent", workflow=agent_workflow, environment=None, thread=thread, task_at_hand=taskObj, agent_model=model_name, client_handler=client_handler, TRAVERSE_MAX_DEPTH=TRAVERSE_MAX_DEPTH)
    graph_state.agent = agent 
    thread.agent = agent 
    return agent 
    