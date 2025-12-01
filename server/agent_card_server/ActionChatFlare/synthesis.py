import os

from chatflare.graph.action import BaseAction
from chatflare.graph.base import GraphState, GraphTraverseThread
from chatflare.prompt.base import PromptTemplate
from chatflare.tracker.base import Blob, Commit
from chatflare.core.llm_wrapper import BaseChain
from chatflare.model.openai import ChatOpenAI
from chatflare.model.llama_bedrock import LlamaBedrock
from chatflare.model.qwen import QwenLLM

from agent_card_server.Output.print import log_action, log_action_res, COLOR_INCLUDE, COLOR_EXCLUDE, log_errors

import json
import copy

from openai import api_key

SUMMARY_PROMPT_PREFIX = """You are a research agent specialized in summarizing and synthesizing findings for a biomedical systematic review report. Your task involves working with a set of paper summaries that have been generated from various articles.
All of these summaries are related to this specific research question "{{query}}".

Given a new paper summary, your job is to review a list of summaries for other papers and also intermediate synthesis, then identify one that is most relevant to this new paper.
Once identified, if the most relevant one is another paper summary, you will synthesize the summary of new paper and the identified paper together into an intermediate synthesis; if the identified one is an intermediate synthesis, you will integrate the findings (summary) of the new paper into the identified synthesis, and ensure the updated synthesis presents key information and insights in a clear and coherent manner, organized as a report with findings, analysis, conclusions, and evidence with citations.
If there are no relevant paper summaries and no relevant intermediate synthesis, clearly state this and explain why.

It’s important to note:
- If there are no relevant summaries, clearly state this and explain why.
- When identifying relevant summaries or intermediate synthesis, please return the ID and indicate the degree of relevance.
- When synthesizing, only present information that can be supported by the provided summaries and intermediate synthesis.
- Include citations in the specified format <citation>citation_number</citation> where "citation_number" corresponds to the ID of the relevant paper summary. If the information comes from the current paper summary, cite it as <citation>{{current_summary_index}}</citation>. 

Here is the summary of the new paper, which is wrapped between a pair of triple backticks: 
```
{{current_summary_index}}:{{paper_summary}}
```

Here are the summaries of other papers and intermediate synthesis that you need to review and compare with the above summary. There may not be any relevant summaries or synthesis in the list, and each summary is given in the format of "ID: Summary" and all summaries are wrapped between a pair of triple backticks:
```
{{previous_summaries}}
```

{{user_instructions_potentially_related}}
Begin by integrating user's research question "{{query}}" and instructions (feedback) into your evaluation criteria. If there are no additional instructions or if the instructions or the feedbacks you considered is irrelevant to the summary task, you can ignore them. 
Review the listed summaries to determine which, if any, align most closely with Summary A in the context of the research question. 
If relevant summaries are identified, synthesize Summary A with these to craft a new, integrated summary. 
If no relevant summaries are found, clearly explain this outcome. Include the IDs of the summaries you deem most relevant, providing detailed reasoning for your selections and how they relate to the research question. 

You may structure the synthesized summary as follows if no additional instructions are provided: (in HTML format, each section title should be in bold and followed by a colon, and the content should be in plain text format not `h`):
1. **Introduction**: Introduce the research question and the broader context. Clearly state what the question aims to address and illustrate its significance. 
2. **Study Design**: Describe how the included summaries were selected and any criteria for inclusion or exclusion of studies. Do not mention any search strategy. Instead, highlight the inclusion-exclusion criteria utilized to determine which studies are integrated. Mention that the following inclusion-exclusion criteria were applied: {{inclusion_exclusion_criteria}}.
3. **Key Findings**: Present the core insights and findings from all evidences in a cohesive manner. Integrate the findings, highlighting similarities, differences, patterns, and contradictions. Ensure every piece of integrated information is properly cited, and similar as well as contradictory findings are clearly summarized.  
4. **Conclusion**: Summarize the overall conclusions drawn from the integrated findings. Analyze the findings to discuss their implications, any patterns or trends, and how they address (or fail to address) the research question.
5. **Discussion**: Discuss the strength of the evidence presented, highlight any gaps or missing information, note limitations, and suggest directions for further research or areas where additional studies might be needed. Also, indicate if some aspects could not be fully addressed given the provided summaries and may require consulting the full articles.

In the meanwhile, cherish the following summarization requirement:
{{summarization_requirement}}

Make sure to cite evidence in the form of <citation>citation_number</citation> at the end of sentence where information from the relevant paper. 
If the information comes from the new paper, cite it with the its index {{current_summary_index}}. Include citations in the form of <citation>citation_number</citation> at the end of sentences where information from the relevant summaries is used. 
When integrate new paper summary into an intermediate synthesis, ensure that the new synthesis integrate the new paper and cite as <citation>{{current_summary_index}}</citation> at the end of sentences where information from the new paper is used. In the meanwhile, ensure all the existing citations are properly integrated and cited in the updated synthesis. Do not cite synthesis' ID, instead stick to the original evidence's citations.
Make sure all the evidence are properly integrated and cited in the generated or updated synthesis.

Be comprehensive and thorough in presenting the integrated synthesis. Use plain text for the content, but retain the HTML tags for each section. Each section should start with a bold title as per the instructions. Avoid overly brief or cursory explanations.
If no relevant summaries are identified, clearly explain this outcome and provide reasons. In that case, the synthesized summary should still follow the same structure but state there are no relevant external findings.

Use information only from the summaries provided and DO NOT introduce or guess new information!

{RESPONSE_FORMAT}
"""

SUMMARY_RESPONSE_FORMAT = """
Respond with a JSON object formatted in the \
following schema:
```json
{{  
    "identified_relevant_summaries": str[] // list of ids (the number at beginning of each summary and before `:`) or empty list if no similar/relevant articles are identifed
    "reasoning": str, //reasoning for why the identified summaries are relevant to Summary A, if any, or the reason for not identifying any relevant summaries
    "synthesized_summary": str, //the new synthesis or updated synthesis that integrates key information and insights from the new paper and the identified relevant summaries, including citations as <citation>citation_number</citation> at the end of sentences where information from the relevant summaries is used
    "thought": str, //your overall understanding of question `{query}` so far, not just from this paper but also your previous findings. Do not include irrelevant information here. Double check whether the all the evidence (cited evidence) from the identified relevant summaries are properly integrated and cited in the generated synthesis.  
}}
```
"""


def create_synthesize_chain(model_name='gpt-4o-mini'):
    prompt_prefix = SUMMARY_PROMPT_PREFIX.\
        format(RESPONSE_FORMAT=SUMMARY_RESPONSE_FORMAT)
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


class SynthesizeAction(BaseAction):
    def __init__(self, action_name="synthesis", model_name: str=None):
        if model_name is None: 
            runnable = create_synthesize_chain()
        else: 
            runnable = create_synthesize_chain(model_name=model_name)
        super().__init__(action_name, runnable)

    def switch_runnable_model(self, model_name: str):
        self.runnable = create_synthesize_chain(model_name=model_name)
    
    async def arun(self, thread: GraphTraverseThread):
        """

        """
        
        log_action("synthesis", len(thread.branch.commits)-1)

        if len(thread.graph_state.cached_work['papers_to_synthesize']) == 0 or len(thread.graph_state.memory.docstore._dict) < 2:
            log_errors(
                "No working summary found in memory or not enough summaries to summarize")
            return

        if len(thread.graph_state.cached_work['papers_to_synthesize']) > 2:
            log_errors(
                "More then 2 papers find need to get synthesized, check reason!")
        paper_to_synthesize_id = thread.graph_state.cached_work['papers_to_synthesize'][-1]
        last_paper_summary_doc = thread.graph_state.memory.docstore._dict[paper_to_synthesize_id]

        previous_summaries_raw = thread.graph_state.memory.associate_in_memory(
            last_paper_summary_doc.page_content)

        inclusion_exclusion_criteria = thread.task.in_criteria
        summarization_requirement = thread.task.summarization_requirement or "No additional summarization requirement."
        previous_summaries_dict = {
            doc.docstore_id: doc for doc, score in previous_summaries_raw if doc.docstore_id != last_paper_summary_doc.docstore_id}

        previous_summaries = "\n".join(
            [f"{doc.docstore_id}: ({'paper-summary' if doc.children is None or len(doc.children)==0 else 'intermediate-synthesis'}){doc.page_content}" for idx,
                (doc, score) in enumerate(previous_summaries_raw) if doc.docstore_id != last_paper_summary_doc.docstore_id]
        )

        user_instructions_potentially_related = ""
        if thread.graph_state.memory.inspiration_conversation_history:
            user_instructions_potentially_related = f"Your thought from previous discussion/feedback from user (may not relevant to current synthesize): {thread.graph_state.memory.inspiration_conversation_history}"
        else:
            user_instructions_potentially_related = "No additional instructions or feedback from user."

        thread.client_handler.emitAgentWorkingProgress({
                "agent_id": thread.agent.agent_id,
                "working_status": "start-synthesis",
                "target_paper": last_paper_summary_doc.metadata,
                "summary": last_paper_summary_doc.page_content,
                "branch": thread.branch.id,
                "article_processing_history": thread.article_processing_history
        }) 
        
        output = await self.runnable.apredict(
            paper_summary=last_paper_summary_doc.page_content,
            previous_summaries=previous_summaries,
            query=thread.task.research_query,
            user_instructions_potentially_related=user_instructions_potentially_related,
            current_summary_index=paper_to_synthesize_id,
            inclusion_exclusion_criteria = inclusion_exclusion_criteria, 
            summarization_requirement = summarization_requirement
        )

        self.sync_with_thread(output, last_paper_summary_doc,
                            previous_summaries_dict, thread)
        return output
        # except Exception as e:
        #     log_errors(f"Error in synthesis: {e}")
        #     return None

    def run(self, thread: GraphTraverseThread):
        """

        """
        log_action("synthesis", len(thread.branch.commits)-1)

        if len(thread.graph_state.cached_work['papers_to_read']) == 0 or len(thread.graph_state.memory.docstore._dict) < 2:
            log_errors(
                "No working summary found in memory or not enough summaries to summarize")
            return

        if len(thread.graph_state.cached_work['papers_to_synthesize']) > 2:
            log_errors(
                "More then 2 papers find need to get synthesized, check reason!")
        paper_to_synthesize_id = thread.graph_state.cached_work['papers_to_synthesize'][-1]
        thread.graph_state.cached_work['papers_to_synthesize'] = []
        last_paper_summary_doc = thread.graph_state.memory.docstore._dict[paper_to_synthesize_id]

        previous_summaries_raw = thread.graph_state.memory.associate_in_memory(
            last_paper_summary_doc.page_content)
        
        for doc, score in previous_summaries_raw:
            if doc.docstore_id == last_paper_summary_doc.docstore_id:
                previous_summaries_raw.remove((doc, score))
                break

        previous_summaries_dict = {
            doc.docstore_id: doc for doc, score in previous_summaries_raw}

        previous_summaries = "\n".join(
            [f"{doc.docstore_id}: {doc.page_content}" for idx,
                (doc, score) in enumerate(previous_summaries_raw)]
        )

        user_instructions_potentially_related = ""
        if thread.graph_state.memory.inspiration_conversation_history:
            user_instructions_potentially_related = f"Your thought from previous discussion/feedback from user (may not relevant to current synthesize): {thread.graph_state.memory.inspiration_conversation_history}"
        else:
            user_instructions_potentially_related = "No additional instructions or feedback from user."

        output = self.runnable.apredict(
            paper_summary=last_paper_summary_doc.page_content,
            previous_summaries=previous_summaries,
            query=thread.task.research_query,
            user_instructions_potentially_related=user_instructions_potentially_related,
            current_summary_index=paper_to_synthesize_id)

        self.sync_with_thread(output, last_paper_summary_doc,
                              previous_summaries_dict, thread)
        if self.runnable.JSON_MODE:
            return json.loads(output)
        return output


    def sync_with_thread(self, output, last_paper_summary_doc, previous_summaries_dict, thread):
        """
        """
        identified_relevant_summaries = output.get(
            "identified_relevant_summaries", [])
        reasoning = output.get("reasoning", "")
        thought = output.get("thought", "")
        synthesized_summary = output.get("synthesized_summary", "")
        if len(identified_relevant_summaries) > 0:
            log_action_res(
                f"Synthesized summary: {synthesized_summary}")
            log_action_res(
                f"Identified relevant summaries: {identified_relevant_summaries}")

        else:
            log_action_res(
                f"No relevant summaries identified for synthesis: {last_paper_summary_doc.docstore_id}\n Reasoning: {reasoning}")
            
            thread.client_handler.emitAgentWorkingProgress({
                    "agent_id": thread.agent.agent_id,
                    "working_status": "end-synthesis",
                    "synthesized_summary": synthesized_summary,
                    "hierarchy": thread.graph_state.memory.output_memories_hierarchy(), 
                    "branch": thread.branch.id,
            })
        synthesize_result_obj = {
                "agentId": thread.agent.agent_id,
                "output": output,
                "content": synthesized_summary if len(identified_relevant_summaries) > 0 else "not included",
                "meta": {
                    "paper_id": last_paper_summary_doc.metadata.get("paper_id", ""),
                    "children": [previous_summaries_dict[idx].docstore_id for idx in identified_relevant_summaries] + [last_paper_summary_doc.docstore_id], 
                    "thought": thought
                },
                "update_longterm_memory": len(identified_relevant_summaries) > 0,
                "agent_current_position": thread.graph_state.agent_current_position, 
                "cached_work": copy.deepcopy(thread.graph_state.cached_work)
            }
        thread.graph_state.cached_work['papers_to_synthesize'] = []
        synthesis_result_blob = Blob("synthesis", synthesize_result_obj)
        commit = Commit.from_blobs([synthesis_result_blob])
        thread.add_commit(commit)
        
        thread.client_handler.emitAgentWorkingProgress({
                    "agent_id": thread.agent.agent_id,
                    "working_status": "end-synthesis",
                    "synthesized_summary": synthesized_summary,
                    "hierarchy": thread.graph_state.memory.output_memories_hierarchy(),
                    "branch": thread.branch.id,
                    "article_processing_history": thread.article_processing_history
            })


        