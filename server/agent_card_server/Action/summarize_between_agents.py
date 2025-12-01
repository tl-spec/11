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
Given a set of summaries (let's call it Summary A, B, C, D, E, etc.), your job is to review these previously generated summaries (Summary A, B, C, D, E, etc.) and synthesize them into a comprehensive report in a clear and coherent manner, organized as a report with findings, analysis, and conclusions.

Here are all previously generated summaries (Summary A, B, C, D, E, etc.) that you need to review and synthesis. Each summary is given in the format of "ID: Summary" and all summaries are wrapped between a pair of triple backticks:
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
Ensure to cite the original sources. If an identified relevant summary includes citations to original summaries, cite those original summaries instead. If the information is a conclusion or does not have an original citation, cite the relevant summary number. 
For example, if information is derived from Summary 0 which cites Summary 123, the citation should appear as <citation>123</citation>. If a conclusion is drawn from Summary 1, the citation should appear as <citation>1</citation>. 

Use information only from the summaries provided and DO NOT introduce or guess new information!

{RESPONSE_FORMAT}
"""

SUMMARY_RESPONSE_FORMAT = """
When responding use a markdown code snippet with a JSON object formatted in the \
following schema:
```json
{{  
    "synthesized_summary": str, //the new, comprehensive summary that integrates key information and insights from Summary A and the identified relevant summaries, including citations as <citation>citation_number</citation> (MUST use this format!!!) at the end of sentences where information from the relevant summaries is used
}}
```

### Example
To provide a clear understanding, here's an example of what the JSON output might look like:
#### Summaries

**Previously generated summaries:**
"-1: Recent studies suggest a complex relationship between genotype variations and infection rates." 
"0: Genotype X shows higher infection rates compared to genotype Y <citation>123</citation>."
"1: Environmental factors also significantly influence infection rates."

#### JSON Output
```json
{{
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
        "previous_summaries", "query", "user_instructions_potentially_related",
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
    async def act(self, query: str, summaries: List[Document], client_handler, env: Any, globalCardId: str, ): 
        # log_action("summarize_between_agents")
        
            
        if summaries is None or len(summaries) < 2: 
            log_errors("No enough summaries found between agents to summarize")
            return
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        summarize_chain = LLMChain(llm=llm, prompt=self.prompt)

        # previous_summaries_raw = agent.memory.associate_in_memory(paper_summary)
        # remove the current working summary from the list of previous summaries
        # for doc, score in previous_summaries_raw:
        #     if doc.docstore_id == agent.memory.latest_summary.docstore_id:
        #         previous_summaries_raw.remove((doc, score))
        #         break
        
        previous_summaries_dict = {doc.docstore_id: doc for doc in summaries}

        previous_summaries = "\n".join(
            [f"{doc.docstore_id}: {doc.page_content}" for doc in summaries]
        )

        user_instructions_potentially_related = ""
        # if agent.inspiration_conversation_history and len(inspiration_conversation_history) > 0: 
        #     user_instructions_potentially_related = f"Your thought from previous discussion/feedback from user (may not relevant to current synthesize): {agent.inspiration_conversation_history}"
        # else:
        user_instructions_potentially_related = "No additional instructions or feedback from user."

        if client_handler: 
            client_handler.emitOverallSummarizationProgress({
                "working_status": "start-summarize",
            })

        print("before summarization between agents")
        raw_response = await summarize_chain.apredict(
            previous_summaries=previous_summaries, query=query, user_instructions_potentially_related=user_instructions_potentially_related
        )
        print("after summarization between agents")
        parsed_response = parse_json_markdown(raw_response)

        synthesized_summary = parsed_response["synthesized_summary"]
        log_action_res(
            f"Synthesized summary: {synthesized_summary}")
        
        env.summary_between_agents = {
            "synthesized_summary": synthesized_summary,
            "source_summaries": summaries
        }
        
        if client_handler: 
            client_handler.emitOverallSummarizationProgress({
                "globalCardId": globalCardId, 
                "working_status": "end-summarize",
                "synthesized_summary": synthesized_summary,
            })
        return parsed_response
        







        

