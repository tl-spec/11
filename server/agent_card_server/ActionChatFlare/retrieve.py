import os

from chatflare.graph.action import BaseAction
from chatflare.graph.base import GraphState, GraphTraverseThread
from chatflare.prompt.base import PromptTemplate
from chatflare.tracker.base import Blob, Commit 
from chatflare.core.llm_wrapper import BaseChain
from chatflare.model.openai import ChatOpenAI
from chatflare.model.llama_bedrock import LlamaBedrock
from chatflare.model.qwen import QwenLLM

from agent_card_server.Output.print import log_action, log_action_res, COLOR_INCLUDE, COLOR_EXCLUDE

import json 
import copy

RETRIEVE_PROMPT_PREFIX = """You are an research agent designed to help human perform an literature review tasks from a set of documents. 
You are give a task to review information of the given research question: {{query}}.
You are given a set of documents to read and summarize, but at the beginning, you just need to decide which paper you want to read on this iteration based on their title and the begining of their abstract. 
{{paper_already_read}}
{{findings_so_far}}
The paper you have access at this iterations are: 
{{available_papers}}
You need to identify whether they are directly related to the research question `{{query}}`. 
{{inclusion_criteria}}
Then you can choose to read one or multiple of the paper by giving the index of paper if you think they are directly relevant. SKIP them if you think the given papers are not directly relevant to the question `{{query}}`. 
Before make selection, you need to first generate a thought on why you make such a selection to read the selected papers and skip the rest. 
Consider that the number of papers you can read is limited, you need to make a wise decision on which paper to read. if you feel that one or the papers give to you contains redundant information based on what you already read, you can choose to skip it.
{{inspiration_conversation_history}}

{RESPONSE_FORMAT}
"""


RETRIEVAL_RESPONSE_FORMAT = """
Return a JSON object formatted in the following schema and nothing else:
```json
{{   
    "thought": str, //your thought on why you make such a selection or `skip` them given the research question {query}
    "selected_papers": List[str], //list of paper_indexs (str) or string 'skip' (instead of list) if none of them is directly relevant to {query}
}}
```
"""


def create_retrieve_chain(model_name='gpt-4o-mini'):
    prompt_prefix = RETRIEVE_PROMPT_PREFIX.\
        format(RESPONSE_FORMAT=RETRIEVAL_RESPONSE_FORMAT)
    prompt = PromptTemplate(template=prompt_prefix)
    if "llama" in model_name:
        model = LlamaBedrock(model_name=model_name)
    elif 'gpt' in model_name:
        model = ChatOpenAI(model_name=model_name)
    elif 'qwen' in model_name:
        model=QwenLLM(model_name=model_name,
                      api_key=os.getenv("QWEN_API_KEY"),
                      base_url=os.getenv("QWEN_BASE_URL"))
    else: 
        model = ChatOpenAI(model_name=model_name)
    chain = BaseChain(model, prompt, json_mode=True)
    return chain


class RetrieveAction(BaseAction): 
    def __init__(self, action_name="retrieve", model_name: str=None):        
        if model_name is None: 
            runnable = create_retrieve_chain()
        else: 
            runnable = create_retrieve_chain(model_name=model_name)
        super().__init__(action_name, runnable)
    
    def switch_runnable_model(self, model_name: str):
        self.runnable = create_retrieve_chain(model_name=model_name)
    
    async def arun(self, thread: GraphTraverseThread):
        log_action("retrieve", len(thread.branch.commits)-1)
        query = thread.task.detailed_research_query
        paper_already_read = "You haven't read any paper yet" 
        if len(thread.graph_state.paper_visited_in_commits) > 0: 
            documents_visited = set()
            for commit, docs in thread.graph_state.paper_visited_in_commits.items(): 
                documents_visited.update(docs)
            paper_already_read = f"You have ready {len(documents_visited)} papers so far\n"
        print("start retrieve-1")
        findings_so_far = ""
        if thread.graph_state.memory.working_memory and len(thread.graph_state.memory.working_memory) > 0:
            findings_so_far = f"Your findings so far: {thread.graph_state.memory.working_memory}"
        inspiration_conversation_history = ""
        if thread.graph_state.memory.inspiration_conversation_history and len(thread.graph_state.memory.inspiration_conversation_history) > 0:
            inspiration_conversation_history = f"Based on your previous conversation with human expert, you have the following inspiration: \n{thread.graph_state.memory.inspiration_conversation_history} Make sure your action reflect on this inspiration.\n"
        inclusion_criteria = ""
        if thread.task.in_criteria:
            inclusion_criteria = f"Inclusion criteria: {thread.task.in_criteria}"
        print("start retrieve-2")
        current_reception_field = thread.graph_state.environment.get_real_agent_reception_field(thread)
        if len(current_reception_field) == 0:
            self.sync_with_thread_after_all_visited(thread)
            return 
            
        available_papers = "\n".join(
            [f"{idx}: {doc.page_content}" for idx, doc in enumerate(current_reception_field)])

        _available_papers_id_to_doc_for_emit = {
            str(idx): {
                "metadata": doc.metadata,
                "abstract": doc.metadata.get("AB", "None"),
                "authors": doc.metadata.get("AU", []),
                "id": doc.docstore_id,
            } \
            for idx, doc in enumerate(current_reception_field)
        }
        print("start retrieve-3")
        tmp_article_processing_history = thread.article_processing_history 
        for doc in _available_papers_id_to_doc_for_emit.values(): 
            tmp_article_processing_history[doc["id"]] = "visited"
        thread.client_handler.emitAgentWorkingProgress({
            "agent_id": thread.agent.agent_id,
            "working_status": "start-retrieve",
            "reception_field": list(_available_papers_id_to_doc_for_emit.values()), 
            "branch": thread.branch.id,
            "article_processing_history": tmp_article_processing_history
        })
        print("start retrieve-5")
        output = await self.runnable.apredict(query=query, paper_already_read=paper_already_read, findings_so_far=findings_so_far, inspiration_conversation_history=inspiration_conversation_history, available_papers=available_papers, inclusion_criteria=inclusion_criteria)
        self.sync_with_thread(output, current_reception_field, _available_papers_id_to_doc_for_emit, thread)
            
        return output

    def run(self, thread: GraphTraverseThread):
        query = thread.task.research_query
        paper_already_read = "You haven't read any paper yet" 
        if len(thread.graph_state.paper_visited_in_commits) > 0: 
            documents_visited = set()
            for commit, docs in thread.graph_state.paper_visited_in_commits.items(): 
                documents_visited.update(docs)
            paper_already_read = f"You have ready {len(documents_visited)} papers so far\n"
        
        findings_so_far = ""
        if thread.graph_state.memory.working_memory and len(thread.graph_state.memory.working_memory) > 0:
            findings_so_far = f"Your findings so far: {thread.graph_state.memory.working_memory}"
        
        inspiration_conversation_history = ""
        if thread.graph_state.memory.inspiration_conversation_history and len(thread.graph_state.memory.inspiration_conversation_history) > 0:
            inspiration_conversation_history = f"Based on your previous conversation with human expert, you have the following inspiration: \n{thread.graph_state.memory.inspiration_conversation_history} Make sure your action reflect on this inspiration.\n"
        
        current_reception_field = thread.graph_state.environment.get_real_agent_reception_field(thread)

        if len(current_reception_field) == 0:
            self.sync_with_thread_after_all_visited(thread)
            return 
        
        available_papers = "\n".join(
            [f"{idx}: {doc.page_content}" for idx, doc in enumerate(current_reception_field)])
        
        inclusion_criteria = ""
        if thread.task.in_criteria:
            inclusion_criteria = f"Inclusion criteria: {thread.task.in_criteria}"
             
        output = self.runnable.predict(query=query, paper_already_read=paper_already_read, findings_so_far=findings_so_far, inspiration_conversation_history=inspiration_conversation_history, available_papers=available_papers, inclusion_criteria=inclusion_criteria)
        self.sync_with_thread(output, current_reception_field, thread)
        if self.runnable.JSON_MODE:
            return json.loads(output)
        return output

    
    def sync_with_thread_after_all_visited(self, thread: GraphTraverseThread): 
        thread.graph_state.finish_all_articles = True
        retrieve_result_obj = {
            "output": {"result": "finish_all_articles"}, 
            "content": "Finish all articles", 
            "meta": {
            }, 
            "update_longterm_memory": False,
            "agent_current_position": thread.graph_state.agent_current_position
        }
        thread.client_handler.emitAgentWorkingProgress({
            "agent_id": thread.agent.agent_id,
            "working_status": "failed-retrieve-out-of-paper",
            "branch": thread.branch.id,
        })
        retrieval_result_blob = Blob("retrieve", retrieve_result_obj)
        commit = Commit.from_blobs([retrieval_result_blob])
        thread.add_commit(commit)
        return 
        
    def sync_with_thread(self, output, reception_field_papers, _available_papers_id_to_doc_for_emit, thread): 
        
        thought = output["thought"]
        selected_papers = output["selected_papers"]
        
        ## 1. check if the output has papers to read, if so, add papers to cached work
        if output["selected_papers"] != "skip":
            thread.graph_state.cached_work["papers_to_read"] = [reception_field_papers[int(idx)] for idx in selected_papers]
            log_action_res(f"Selected {len(selected_papers)} papers")
            log_action_res(f"Reason of selection: {thought}")

            selected_papers_for_emit = [_available_papers_id_to_doc_for_emit[str(
                idx)] for idx in selected_papers]
            
        else: 
            log_action_res(f"No paper selected with the reason: {thought}")

                    
        thread.graph_state.latest_output = output       
        thread.graph_state.memory.working_memory = thought
        ## 2. submit a commit to thread, and thread need to update the graph state
        retrieve_result_obj = {
            "agentId": thread.agent.agent_id,
            "output": output, 
            "content": None, 
            "meta": {
                "reception_field_papers": [doc.docstore_id for doc in reception_field_papers], 
                "thought": thought, 
            }, 
            "update_longterm_memory": False,
            "agent_current_position": reception_field_papers[-1].docstore_id if output["selected_papers"] == "skip" else reception_field_papers[int(selected_papers[0])].docstore_id, 
            "cached_work": copy.deepcopy(thread.graph_state.cached_work),
        }
        
        thread.graph_state.agent_current_position = reception_field_papers[-1].docstore_id if output["selected_papers"] == "skip" else reception_field_papers[int(selected_papers[0])].docstore_id
        retrieval_result_blob = Blob("retrieve", retrieve_result_obj)
        commit = Commit.from_blobs([retrieval_result_blob])
        thread.add_commit(commit) 
        ## 3. adjust paper visited in thread.graph_state
        thread.graph_state.paper_visited_in_commits[commit.id] = [doc.docstore_id for doc in reception_field_papers]
        if output["selected_papers"] != "skip":
            thread.client_handler.emitAgentWorkingProgress({
                    "agent_id": thread.agent.agent_id,
                    "working_status": "finish-retrieve",
                    "paper_selected": selected_papers_for_emit,  # skip | list of paper index
                    "thought": thought, 
                    "branch": thread.branch.id,
                    "article_processing_history": thread.article_processing_history 
            })
            
        else: 

            thread.client_handler.emitAgentWorkingProgress({
                    "agent_id": thread.agent.agent_id,
                    "working_status": "finished-retrieve",
                    "paper_selected": [],  # skip | list of paper index
                    "thought": thought, 
                    "branch": thread.branch.id,
                    "article_processing_history": thread.article_processing_history 
            })
