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


OVERALL_SUMMARY_PROMPT_PREFIX = """You are a research agent tasked with creating a comprehensive summary that integrates both synthesized findings from similar articles and important findings from articles not directly related to the synthesized content. These findings, while varied, are all relevant to the user's overarching research question {{query}}. 
Your objective is to weave together these elements into a cohesive report that not only presents a unified summary but also critically examines the material, especially highlighting any conflicting findings that emerge from the synthesis and the broader research context.
In undertaking this task, you may be guided by specific summary requirements provided by the user. If no such guidelines are given, you are expected to apply your professional judgment and principles to structure the report in a manner that best addresses the research question, ensuring clarity, coherence, and comprehensive coverage of the findings."

Here are the summaries that you need to work with, all summaries are wrapped between a pair of triple backticks:
```
{{previous_summaries}}
```

{{user_instructions_potentially_related}}

Begin by integrating user's research question "{{query}}" and instructions (feedback) into your evaluation criteria. 
Your primary task is to create a comprehensive report that integrates the synthesized findings with the important non-synthesized findings, maintaining a logical flow and coherence throughout. In your synthesis:
- Begin by outlining the main insights from the synthesized findings.
- Clearly identify and discuss any conflicting findings or perspectives that emerge from the synthesis if there are any, providing a critical examination of these discrepancies.
- If specific summary requirements are provided, ensure your report aligns with these guidelines. In the absence of such instructions, utilize your professional judgment to determine the most effective structure and emphasis for the report.

{RESPONSE_FORMAT}
"""

OVERALL_SUMMARY_RESPONSE_FORMAT = """
When responding use a markdown code snippet with a JSON object formatted in the \
following schema:
```json
{{  
    "summary_report": str, //the comprehensive summary that integrates key information and insights from list of summaries. Your final report should present a nuanced, well-rounded exploration of the research question, emphasizing the convergence and divergence within the findings and offering insights that advance the user's understanding of the topic.
}}
```
Return only the makrdown and Nothing Else!
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


class OverallSummarizeAction(BaseAction):
    action_name: str = "summarize" 
    prompt_prefix: str = OVERALL_SUMMARY_PROMPT_PREFIX.format(
        RESPONSE_FORMAT = OVERALL_SUMMARY_RESPONSE_FORMAT)
    
    prompt: PromptTemplate = PromptTemplate(template=prompt_prefix, input_variables=[
        "previous_summaries", "query", "user_instructions_potentially_related"
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
        log_action("overall_summarize_report", agent.current_iter)
        
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        summarize_chain = LLMChain(llm=llm, prompt=self.prompt)
        
        previous_summaries_raw = agent.memory.get_inner_memories()
        if len(previous_summaries_raw) == 0:
            log_errors("No enough summaries to report")
            return
        
        previous_summaries = "\n".join([f"{idx}: {doc.page_content}" for idx, doc in enumerate(previous_summaries_raw)])


        user_instructions_potentially_related = "No additional instructions or feedback from user."
        if agent.report_instructions and len(agent.report_instructions) > 0: 
            user_instructions_potentially_related = f"User's instructions and requirement for summarization report: {agent.report_instructions}"

        raw_response = await summarize_chain.apredict(
            previous_summaries=previous_summaries, query=query, user_instructions_potentially_related=user_instructions_potentially_related
        )
        print(raw_response)
        parsed_response = parse_json_markdown(raw_response)


        summary_report = parsed_response["summary_report"]


        id_in_list = agent.memory.add_memory([summary_report], [{
                "children_summaries": [doc.docstore_id for doc in previous_summaries_raw]
        }], parent=[None], children=[[doc for doc in previous_summaries_raw]])

        id_in_list = id_in_list[0]

        log_action_res(f"overall_summarize_report: {summary_report} \n Overall report id: {id_in_list}")
        
        for doc in previous_summaries_raw:
            doc.parent = agent.memory.docstore._dict[id_in_list]
        
        agent.memory.overall_report = agent.memory.docstore._dict[id_in_list]
        agent.action_sequence_done.append({
                "time_stamp": agent.current_iter,
                "action": "overall_summarize",
                "findings": f"Your overall report has been successfully generated and stored in memory: {agent.memory.overall_report.page_content}"
        })
        return parsed_response







        

