import os

from chatflare.graph.action import BaseAction
from chatflare.graph.base import GraphState, GraphTraverseThread
from chatflare.prompt.base import PromptTemplate
from chatflare.tracker.base import Blob, Commit
from chatflare.core.llm_wrapper import BaseChain
from chatflare.model.openai import ChatOpenAI
from chatflare.model.llama_bedrock import LlamaBedrock
from chatflare.model.qwen import QwenLLM

from agent_card_server.Output.print import log_action, log_action_res, log_errors, COLOR_INCLUDE, COLOR_EXCLUDE

import json


REFLECT_ON_HUMAN_QUESTION_INSTRUCTION = """You are an research agent designed to help human expert perform a systematic review tasks from a set of documents.
You are give a task to review information of the given research question: {{query}}. 
{{include_exclude_criteria}}

You have already read several papers and got some findings, now the human expert discusses with some questions based on your findings and give you further guidance on your work. 
{{findings_so_far}} (Note: The findings listed here represent an overall insights.)
{{conversation_history}}
Based on the question, instruction and possible critics from human experts, you need to reflect on your work so far and give a thought on how to improve your work in the next iteration to match human's expectation. 
If so far, the human experts only have positive feedback on your work, or questions about your findings so far you can just say that you will keep doing what you are doing.
However, if human have instructions, clarifications or critics on your work, you need to reflect on your work and give a thought on how to improve your work in the next iteration to match human's expectation.
{RESPONSE_FORMAT}
"""

REFLECT_RESPONSE_FORMAT = """
When responding use a markdown code snippet with a JSON object formatted in the \
following schema:
```json
{{   
    "reflection": str, // your overall reflection on how to improve your work in the next iteration to match human's expectation.
    "updates_on_additional_requirement": str, // Based on your reflection, your updates or add additional research focus to the query/research question based on the human expert's feedback question, instruction or critics, return '' (empty string) only if nothing need to changed. 
    "updates_on_criteria": str // based on your reflection, your updates on the inclusion and exclusion criteria based on the human expert's question, instruction or critics. Return '' (empty string) if nothing need to changed.
}}
```
Return only the makrdown and Nothing Else!
"""



DISCUSS_WITH_HUMAN_PROMPT_PREFIX = """You are a research agent designed to assist human experts in performing systematic review tasks from a set of documents. You have been given a task to review information related to the given research question. {{query}} A human expert needs to discuss these findings with you, ask questions based on what you have discovered so far, and provide further guidance on your work. 
{{environment}}
{{paper_already_read}}
{{include_exclude_criteria}}
{{findings_so_far}} 
{{relevant_findings}}
{{conversation_history}}

Please response the human expert's input (questions, instructions or greetings), based on the papers you have already read and your findings so far. Ensure that your responses are supported by the papers you have read and do not include statements that cannot be substantiated by these documents.
Human's input: {{human_expert_question}}
{RESPONSE_FORMAT}
If you think you have enough information to respond to the human expert's question, like if the human experts only have positive feedback on your work, or questions about your findings so far you can just say that you will keep doing what you are doing. you can provide a response directly.
If you need to ask the human expert for clarification or further guidance for research question, inclusion exclusion criteria for the review, you can do so by asking a question in the response.

If you think user have an hint or explicit instruction to have you adjust your inclusion and exclusion stragegy to include more relevant papers, or you need to reflect on the adjusting existing reading path, or adjusting the research question or retrieval focus, return `whether_to_reflect` as True. This is usually because human have instructions, clarifications or critics on your work, you need to reflect on your work and give a thought on how to improve your work in the next iteration to match human's expectation.
If user has explicitly expressed the intent to have you continue your workflow, and you think you already have enough guidance and are ready to continue the retrieval, read and synthesize more papers, please return `whether_to_continue` as True. This is usually because human have clear instructions or intentions to have you continue the workflow. 

"""

DISCUSS_RESPONSE_FORMAT = """
When responding to the human expert's question and instructions, ensure your communication is polite. When you mention specific findings, include a citation (paper's title) next to it in parenthesis. Importantly, if asked about overall findings, directly notify the expert in your response that your current insights form a working memory and do not constitute a comprehensive summary. Assure them that a more detailed and comprehensive summary can be provided with additional time for synthesis.
Responding use a JSON object formatted in the \
following schema:
```json
{{  
    "analysis": str, // Your analysis of the human expert's question or instruction, what you think they are asking for, should you improve your strategy, current research question, should they intend to have you continue or they would like you to pause and answer their question, and how you plan to respond.
    "whether_to_reflect": bool // Whether you need to modify retrieval and synthesis strategies, or adjusting existing reading path. Only return false when this is clearly not necessary for some causal question.
    "whether_to_continue_workflow": bool // Whether user has explicitly ask you to continue the retrieval workflow. Only return true when user has expressed or indicated to continue the workflow and you have been assigned to an environment! Otherwise, ask the user to assign you to an environment.
    "response": str //  Your response to the human expert's question or instruction.
}}
```
"""


def create_discuss_chain(model_name='gpt-4o-mini'):
    prompt_prefix = DISCUSS_WITH_HUMAN_PROMPT_PREFIX.\
        format(RESPONSE_FORMAT=DISCUSS_RESPONSE_FORMAT)
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


def create_reflect_chain(model_name='gpt-4o-mini'):
    prompt_prefix = REFLECT_ON_HUMAN_QUESTION_INSTRUCTION.\
        format(RESPONSE_FORMAT=REFLECT_RESPONSE_FORMAT)
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

    
class DiscussAction(BaseAction): 
    def __init__(self, action_name="discuss", model_name: str=None):
        if model_name is None: 
            runnable = create_discuss_chain()
        else: 
            runnable = create_discuss_chain(model_name=model_name)
        super().__init__(action_name, runnable)
        self.reflection = ReflectAction(model_name=model_name)

    def switch_runnable_model(self, model_name: str):
        self.runnable = create_discuss_chain(model_name=model_name)
        self.reflection.runnable = create_reflect_chain(model_name=model_name)
    
    async def arun(self, thread: GraphTraverseThread):
        # try: 
            # log_action("Discuss", len(thread.branch.commits)-1)
            if len(thread.graph_state.human_cached_instructions) == 0:
                log_errors(
                    "No human instructions found in the conversation history. Please provide human instructions to continue the conversation.")
                return   
            environment = "You haven't been assigned to any environment yet.\n Before you start, you need ask the user to assign you to an environment.\n"
            if thread.graph_state.environment and len(thread.graph_state.environment.literature_bank.docstore._dict) > 0:
                environment = f"You are currently assigned in a copurs environemnt with { len(thread.graph_state.environment.literature_bank.docstore._dict)} papers\n"
            human_expert_question = "\n".join(thread.graph_state.human_cached_instructions)
            thread.graph_state.human_cached_instructions = []
            
            paper_already_read = "You haven't read any paper yet.\n"
            if len(thread.graph_state.memory.index_to_docstore_id) > 0:
                paper_already_read = f"You have read {len(thread.graph_state.memory.get_leaf_memories())} papers so far \n"
                paper_already_read += f"And you findings on the most relevant ones are: \n"
                paper_already_read += "\n".join([doc.page_content for doc in thread.graph_state.memory.get_inner_memories()])

            findings_so_far = ""
            if thread.graph_state.memory.working_memory and len(thread.graph_state.memory.working_memory):
                findings_so_far = f"Your working memory so far: \n {thread.graph_state.memory.working_memory} (Note: The findings listed here represent an overall insights.)\n"
            conversation_history = "" 
            if len(thread.graph_state.conversation_history) > 0: 
                conversation_history = f"Conversation history between you and the human expert are: \n"
                conversation_history += "\n".join(\
                    [f"{chat['role']}: {chat['message']}" for chat in thread.graph_state.conversation_history])
            
            query = ""
            if thread.task.detailed_research_query and len(thread.task.detailed_research_query) > 0:
                query = f"You are tasked to review information related to the given research question: {thread.task.detailed_research_query}."
            else: 
                query = "You have not been given a research question yet. Please ask the human expert for the research question."    
            
            include_exclude_criteria = thread.task.detailed_inclusion_exclusion_criteria

            relevant_findings = ""
            if len(thread.graph_state.memory.index_to_docstore_id) > 0:
                relevant_findings += "Your findings so far from previous actions: \n"
                relevant_findings_raw = thread.graph_state.memory.associate_in_memory(human_expert_question)
                if relevant_findings_raw:
                    relevant_findings = "Based on your previous findings, the most relevant papers are: \n"
                    relevant_findings += "\n".join([f"{doc.page_content}" for doc, score in relevant_findings_raw])

            thread.client_handler.emitAgentWorkingProgress({
                    "agent_id": thread.agent.agent_id,
                    "working_status": "reflecting",
                    "branch": thread.branch.id,
            })
            output = await self.runnable.apredict(query=query, human_expert_question=human_expert_question, environment=environment, paper_already_read=paper_already_read, findings_so_far=findings_so_far, conversation_history=conversation_history, include_exclude_criteria=include_exclude_criteria, relevant_findings=relevant_findings, debug=True)
            
            self.sync_with_thread(output, human_expert_question, thread)
            
            # if output.get("whether_to_reflect", False):
            # reflection_action = ReflectAction() 
            reflection_output = await self.reflection.arun(thread) 
            output["reflection"] = reflection_output.get("reflection", "")
            output["updates_on_criteria"] = reflection_output.get("updates_on_criteria", "")
            output["updates_on_additional_requirement"] = reflection_output.get("updates_on_additional_requirement", "")
                
            return output
        # except Exception as e:
        #     print(f"Error in discuss action: {e}")
        #     return None

    def run(self, thread): 
        """"""
        print("run discuss")
            
        
    def sync_with_thread(self, output, human_expert_question, thread): 
        response = output.get("response", "")
        log_action_res(f"Agent: {response}")
        thread.client_handler.emitAgentWorkingProgress({
                "agent_id": thread.agent.agent_id,
                "working_status": "reflected",
                "response": response,
                "branch": thread.branch.id,
        })
        whether_to_continue_workflow = output.get("whether_to_continue_workflow", False)
        whether_to_reflect = output.get("whether_to_reflect", False)
        thread.graph_state.conversation_history.append({"role": "human", "message": human_expert_question})
        thread.graph_state.conversation_history.append({"role": "agent", "message": response})  
        discuss_result_obj = {
                "output": output,
                "content": response, 
                "meta": {
                    # "thought": response
                },
                "update_longterm_memory": False,
                "agent_current_position": thread.graph_state.agent_current_position
            }
        
        discuss_result_blob = Blob("discuss", discuss_result_obj)
        commit = Commit.from_blobs([discuss_result_blob])
        thread.add_commit(commit)
        
                
class ReflectAction(BaseAction):
    def __init__(self, action_name="reflect", model_name: str=None):
        if model_name is not None:
            runnable = create_reflect_chain(model_name=model_name)
        else: 
            runnable = create_reflect_chain()
        super().__init__(action_name, runnable)

    async def arun(self, thread: GraphTraverseThread):
        # try: 
            log_action("Reflect", len(thread.branch.commits) - 1)

            if len(thread.graph_state.conversation_history) == 0:
                log_errors(
                    "No conversation history found to reflect on. Please provide context for reflection."
                )
                return

            # Prepare variables for reflection chain
            query = thread.task.detailed_research_query

            include_exclude_criteria = "The include and exclude criteria are not provided yet."
            if thread.task.in_criteria:
                include_exclude_criteria = "The include and exclude criteria are: \n"
                include_exclude_criteria += f"{thread.task.in_criteria}\n"            

            conversation_history = "".join(
                [f"{chat['role']}: {chat['message']}\n" for chat in thread.graph_state.conversation_history]
            )
            findings_so_far = ""
            if thread.graph_state.memory.working_memory and len(thread.graph_state.memory.working_memory):
                findings_so_far = f"Your working memory so far: \n{thread.graph_state.memory.working_memory}\n"

            # Call the reflect chain
            output = await self.runnable.apredict(
                query=query,
                include_exclude_criteria=include_exclude_criteria,
                conversation_history=conversation_history,
                findings_so_far=findings_so_far
            )


            self.sync_with_thread(output, thread)
            return output
        
        # except Exception as e:
        #     print(f"Error in reflect action: {e}")
        #     return None

    def run(self, thread): 
        """"""
        print("run reflect")

    def sync_with_thread(self, output, thread):
        reflection = output.get("reflection", "")
        updates_on_criteria = output.get("updates_on_criteria", thread.task.in_criteria)
        updates_on_additional_requirement = output.get(
            "updates_on_additional_requirement", thread.task.user_specified_requirement
        )

        # Log the reflection result
        log_action_res(f"Agent Reflection: {reflection}")
        print(output)
        print(updates_on_criteria)
        print(updates_on_additional_requirement)
        # Update the task criteria and additional requirements
        if updates_on_criteria and len(updates_on_criteria) > 0:
            log_action_res(f"Agent Criteria Update: {updates_on_criteria}")
            thread.agent.update_inclusion_exclusion_critera(updates_on_criteria)
        if updates_on_additional_requirement and len(updates_on_additional_requirement) > 0:
            log_action_res(f"Agent Additional Requirement Update: {updates_on_additional_requirement}")           
            thread.agent.update_user_specified_requirement(updates_on_additional_requirement)
        # Add reflection result to conversation history and thread commits
        thread.graph_state.conversation_history.append(
            {"role": "agent", "message": reflection}
        )
        thread.graph_state.memory.inspiration_conversation_history = reflection

        reflect_result_obj = {
            "agentId": thread.agent.agent_id,
            "output": output,
            "content": reflection,
            "meta": {
                "inspiration_from_conversation": reflection
            },
            "update_longterm_memory": False,
            "agent_current_position": thread.graph_state.agent_current_position
        }

        reflect_result_blob = Blob("reflect", reflect_result_obj)
        commit = Commit.from_blobs([reflect_result_blob])
        thread.add_commit(commit)
