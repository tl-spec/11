from agent_card_server.Action.base import BaseAction
from langchain import LLMChain, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from agent_card_server.Memory.base import MemoryBank
from typing import List, Dict, Optional, Union, Any
from agent_card_server.Agent.base import BaseAgent
from agent_card_server.Output.print import log_action, log_action_res, COLOR_INCLUDE, COLOR_EXCLUDE, log_errors
import tenacity
import asyncio

SUMMARY_PROMPT_PREFIX = """You are a research agent specialized in summarizing and synthesizing information. Your task involves working with a set of summaries that have been generated from various articles.
All of these summaries are related to this specific research question "{{query}}".
Given a new summary (let's call it Summary A), your job is to review a list of previously generated summaries (Summary B, C, D, E, etc.) and identify the ones that are most relevant to Summary A.
Once identified, you will synthesize Summary A with the most similar or relevant summaries from the list to create a new, comprehensive summary. This new summary should integrate the key information and insights from the selected summaries in a clear and coherent manner, organized as a report with findings, analysis, conclusions, and evidence with citations.
It’s important to note that there may not always be relevant summaries; in such cases, this should be clearly stated. Additionally, when you identify similar or relevant summaries, please return the IDs of these summaries with as fine granularity as possible, indicating the degree of relevance or similarity to Summary A. 

Here is the Summary A (the new summary you need to work with) is wrapped between a pair of triple backticks: 
```
{{paper_summary}}
```

Here are the previously generated summaries (Summary B, C, D, E, etc.) that you need to review and compare with the above summary, remember, there may not be any relevant summaries in the list, and each summary is given in the format of "ID: Summary" and all summaries are wrapped between a pair of triple backticks:
```
{{previous_summaries}}
```

{{user_instructions_potentially_related}}
Begin by integrating user's research question "{{query}}" and instructions (feedback) into your evaluation criteria. If there are no additional instructions or if the instructions or the feedbacks you considered is irrelevant to the summary task, you can ignore them. 
Review the listed summaries to determine which, if any, align most closely with Summary A in the context of the research question. 
If relevant summaries are identified, synthesize Summary A with these to craft a new, integrated summary. 
If no relevant summaries are found, clearly explain this outcome. Include the IDs of the summaries you deem most relevant, providing detailed reasoning for your selections and how they relate to the research question. 

When crafting the synthesized summary, include the following sections with each section start with a new paragraph in HTML format, not markdown, each section title should be in bold and followed by a colon, and the content should be in plain text format not `h`. The sections are as follows:
1. **Introduction**: Briefly introduce the research question and the context.
2. **Findings**: Present the key findings from the summaries, integrating insights from Summary A and the relevant summaries.
3. **Analysis**: Analyze the findings, discussing their implications and any patterns or trends observed.
4. **Conclusion**: Summarize the overall conclusions drawn from the analysis.
Make sure to cite the relevant summaries in the form of <citation>citation_number</citation> at the end of sentences where information from the relevant summaries is used. 
Ensure to cite the original sources. If an identified relevant summary includes citations to original summaries, cite those original summaries instead. If the information is a conclusion or does not have an original citation, cite the relevant summary number. If the information comes from the current summary (Summary A), cite it with the its index {{current_summary_index}}. Include citations in the form of <citation>citation_number</citation> at the end of sentences where information from the relevant summaries is used. 
For example, if information is derived from Summary 0 which cites Summary 123, the citation should appear as <citation>123</citation>. If a conclusion is drawn from Summary 1, the citation should appear as <citation>1</citation>. If the information comes from Summary A, the citation should appear as <citation>{{current_summary_index}}</citation>.

Use information only from the summaries provided and DO NOT introduce or guess new information!

{RESPONSE_FORMAT}
"""

SUMMARY_RESPONSE_FORMAT = """
When responding use a markdown code snippet with a JSON object formatted in the \
following schema:
```json
{{  
    "identified_relevant_summaries": str[] // list of ids (the number at beginning of each summary and before `:`) or empty list if no similar/relevant articles are identifed
    "reasoning": str, //reasoning for why the identified summaries are relevant to Summary A, if any, or the reason for not identifying any relevant summaries
    "synthesized_summary": str, //the new, comprehensive summary that integrates key information and insights from Summary A and the identified relevant summaries, including citations as <citation>citation_number</citation> (MUST use this format!!!) at the end of sentences where information from the relevant summaries is used
}}
```

### Example
To provide a clear understanding, here's an example of what the JSON output might look like:
#### Summaries
**Summary A (new summary to work with):**
"Recent studies suggest a complex relationship between genotype variations and infection rates." (index -1 for example)
**Previously generated summaries:**
"0: Genotype X shows higher infection rates compared to genotype Y <citation>123</citation>."
"1: Environmental factors also significantly influence infection rates."

#### JSON Output
```json
{{
    "identified_relevant_summaries": ["0", "1"],
    "reasoning": "Summary 0 discusses the impact of genotype variations on infection rates, which is directly relevant to Summary A. Summary 1 highlights environmental factors affecting infection rates, which provides additional context.",
    "synthesized_summary": "Recent studies suggest a complex relationship between genotype variations and infection rates <citation>-1</citation>. Genotype X shows higher infection rates compared to genotype Y <citation>123</citation>. Environmental factors also significantly influence infection rates <citation>1</citation>."
}}
```
Return only the markdown and Nothing Else!
"""

def parse_json_markdown(json_string: str) -> dict:
    try:
        re
        json
    except:
        import re
        import json
    # Try to find JSON string within triple backticks
    match = re.search(r"```(json)?(.*?)```", json_string, re.DOTALL)

    # If no match found, assume the entire string is a JSON string
    if match is None:
        json_str = json_string
    else:
        # If match found, use the content within the backticks
        json_str = match.group(2)

    # Strip whitespace and newlines from the start and end
    json_str = json_str.strip()

    # Parse the JSON string into a Python dictionary
    parsed = json.loads(json_str)

    return parsed


class SummarizeAction(BaseAction):
    action_name: str = "summarize" 
    prompt_prefix: str = SUMMARY_PROMPT_PREFIX.format(
        RESPONSE_FORMAT=SUMMARY_RESPONSE_FORMAT)
    
    prompt: PromptTemplate = PromptTemplate(template=prompt_prefix, input_variables=[
        "paper_summary", "previous_summaries", "query", "user_instructions_potentially_related", "current_summary_index"
    ])

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(2),
        wait=tenacity.wait_none(),  # No waiting time between retries
        retry=tenacity.retry_if_exception_type(ValueError),
        before_sleep=lambda retry_state: print(
            f"ValueError occurred: {retry_state.outcome.exception()}, retrying..."),
        # Default value when all retries are exhausted
        retry_error_callback=lambda retry_state: 0
    )
    async def act(self, agent: BaseAgent, query: str, scratch_pad: str, working_memory: str, memory: MemoryBank, **kwargs: Any): 
        log_action("summarize", agent.current_iter)
        
            
        if agent.memory.latest_summary is None or len(agent.memory.docstore._dict) < 2: 
            log_errors("No working summary found in memory or not enough summaries to summarize")
            return
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        summarize_chain = LLMChain(llm=llm, prompt=self.prompt)
        paper_summary = agent.memory.latest_summary.page_content
        previous_summaries_raw = agent.memory.associate_in_memory(paper_summary)
        # remove the current working summary from the list of previous summaries
        for doc, score in previous_summaries_raw:
            if doc.docstore_id == agent.memory.latest_summary.docstore_id:
                previous_summaries_raw.remove((doc, score))
                break
        
        previous_summaries_dict = {doc.docstore_id: doc for doc, score in previous_summaries_raw}

        previous_summaries = "\n".join(
            [f"{doc.docstore_id}: {doc.page_content}" for idx, (doc, score) in enumerate(previous_summaries_raw)]
        )
        print(f"working_summary_id: {agent.memory.latest_summary.docstore_id}")
        current_summary_index = agent.memory.latest_summary.docstore_id
        print(f"previous_summaries: {[doc.docstore_id for doc, score in previous_summaries_raw]}")

        user_instructions_potentially_related = ""
        if agent.inspiration_conversation_history and len(inspiration_conversation_history) > 0: 
            user_instructions_potentially_related = f"Your thought from previous discussion/feedback from user (may not relevant to current synthesize): {agent.inspiration_conversation_history}"
        else:
            user_instructions_potentially_related = "No additional instructions or feedback from user."

        if agent.client_handler: 
            agent.client_handler.emitAgentWorkingProgress({
                "agent_id": agent.agent_id,
                "working_status": "start-summarize",
                "target_paper": agent.memory.latest_summary.metadata,
                "summary": agent.memory.latest_summary.page_content,
            })
        print("before summarize_chain.apredict")
        raw_response = await summarize_chain.apredict(
            paper_summary=paper_summary, previous_summaries=previous_summaries, query=query, user_instructions_potentially_related=user_instructions_potentially_related, current_summary_index=current_summary_index
        )
        print("after summarize_chain.apredict")
        print("raw response")
        print(raw_response)
        parsed_response = parse_json_markdown(raw_response)

        identified_relevant_summaries = parsed_response["identified_relevant_summaries"]
        if len(identified_relevant_summaries) == 0:
            log_action_res(
                f"No relevant summaries identified for synthesis: {paper_summary}\n Reasoning: {parsed_response['reasoning']}")
            agent.memory.latest_summary = None 
            if agent.client_handler: 
                agent.client_handler.emitAgentWorkingProgress({
                    "agent_id": agent.agent_id,
                    "working_status": "end-summarize",
                    "synthesized_summary": parsed_response["synthesized_summary"],
                    "hierarchy": agent.memory.output_memories_hierarchy()
                })
            return parsed_response
        else:
            synthesized_summary = parsed_response["synthesized_summary"]
            log_action_res(
                f"Synthesized summary: {synthesized_summary}")
            log_action_res(
                f"Identified relevant summaries: {identified_relevant_summaries}")
            
            id_in_list = agent.memory.add_memory([synthesized_summary], [{
                "children_summaries": [idx for idx in identified_relevant_summaries]
            }], parent=[None], children=[[previous_summaries_dict[idx] for idx in identified_relevant_summaries] + [agent.memory.latest_summary]])

            if id_in_list is None or len(id_in_list) == 0:
                log_errors("Failed to add synthesized summary to memory")
                return
            id_in_list = id_in_list[0]
            print(f"synthesized summary id: {id_in_list}")
            for idx in identified_relevant_summaries:
                doc = previous_summaries_dict[idx]
                doc.parent = agent.memory.docstore._dict[id_in_list]
            agent.memory.latest_summary.parent = agent.memory.docstore._dict[id_in_list]
            agent.memory.latest_summary = None 
            agent.action_sequence_done.append({
                "time_stamp": agent.current_iter,
                "action": "summarize",
                "findings": f"You synthesized a new summary based on the previous summaries. The new summary is: {synthesized_summary}"
            })
            if agent.client_handler: 
                agent.client_handler.emitAgentWorkingProgress({
                    "agent_id": agent.agent_id,
                    "working_status": "end-summarize",
                    "synthesized_summary": synthesized_summary,
                    "hierarchy": agent.memory.output_memories_hierarchy()
                })
            return parsed_response







        

