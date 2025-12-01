from agent_card_server.Action.base import BaseAction#导入基础动作类，用于继承实现自定义动作
from langchain import LLMChain, PromptTemplate #导入LangChain的链和提示模板工具
from langchain_openai import ChatOpenAI#导入OpenAI的聊天模型
from langchain.docstore.document import Document#导入文档存储的文档类
from agent_card_server.Memory.base import MemoryBank#导入记忆库基础类
from typing import List, Dict, Optional, Union, Any#导入类型提示工具
from agent_card_server.Agent.base import BaseAgent#导入代理基础类
from agent_card_server.Output.print import log_action, log_action_res, COLOR_INCLUDE, COLOR_EXCLUDE#导入日志打印工具
import tenacity#导入重试工具
import asyncio

#定义反思人类问题指令的提示词模板
REFLECT_ON_HUMAN_QUESTION_INSTRUCTION = """You are an research agent designed to help human expert perform an literature review tasks from a set of documents.
You are give a task to review information of the given research question: {{query}}. You have already read several papers and got some findings, now the human expert need to discuss with some questions based on your findings and give you further guidance on your work. 
{{action_history}}
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
    "instruction_included": bool // whether the instruction is included in the human expert's question, false if only general questions or positive feedbacks are given. 
    "reflection": str, // your reflection on how to improve your work in the next iteration to match human's expectation.
}}
```
Return only the makrdown and Nothing Else!
"""

# DISCUSS_WITH_HUMAN_PROMPT_PREFIX = """You are an research agent designed to help human expert perform an literature review tasks from a set of documents.
# You are give a task to review information of the given research question: {{query}}. You have already read several papers and got some findings, now the human expert need to discuss with some questions based on your findings and give you further guidance on your work. 
# {{paper_already_read}}
# {{findings_so_far}}
# {{action_history}}
# {{conversation_history}}
# Please answer human expert's question completely based on the paper you have already read and your findings so far, do not give answers or statements that cannot be supported by the papers you have read. 
# {RESPONSE_FORMAT}
# {{human_expert_question}}
# """

# DISUCUSS_RESPONSE_FORMAT = """
# When responed to human expert's question and instructions, make sure it is polite and when you answer with specific findings, make sure to include citation (paper's title) next to it and with parenthesis.
# """
DISCUSS_WITH_HUMAN_PROMPT_PREFIX = """You are a research agent designed to assist human experts in performing literature review tasks from a set of documents. You have been given a task to review information related to the given research question: {{query}}. After reading several papers, you have gathered some preliminary findings. Now, a human expert needs to discuss these findings with you, ask questions based on what you have discovered so far, and provide further guidance on your work. 
{{paper_already_read}}
{{findings_so_far}} (Note: The findings listed here represent initial insights and are part of our working memory. They are not yet a comprehensive summary of all literature reviewed. If a comprehensive synthesis is needed, please explicitly notify the user in your response that this working memory is not comprehensive and that a more detailed summary can be prepared with additional time.)
{{action_history}}
{{conversation_history}}
Please answer the human expert's questions fully, based on the papers you have already read and your findings so far. Ensure that your responses are supported by the papers you have read and do not include statements that cannot be substantiated by these documents.
{RESPONSE_FORMAT}
{{human_expert_question}}
"""

DISCUSS_RESPONSE_FORMAT = """
When responding to the human expert's question and instructions, ensure your communication is polite. When you mention specific findings, include a citation (paper's title) next to it in parenthesis. Importantly, if asked about overall findings, directly notify the expert in your response that your current insights form a working memory and do not constitute a comprehensive summary. Assure them that a more detailed and comprehensive summary can be provided with additional time for synthesis.
"""


RETRIEVE_PROMPT_PREFIX = """You are an research agent designed to help human perform an literature review tasks from a set of documents. 
You are give a task to review information of the given research question: {{query}}.
You are given a set of documents to read and summarize, but at the beginning, you just need to decide which paper you want to read on this iteration based on their title and the begining of their abstract. 
{{paper_already_read}}
{{findings_so_far}}
The paper you have access at this iterations are: 
{{available_papers}}
You need to identify whether they are directly related to the research question `{{query}}`. Then you can choose to read one or multiple of the paper by giving the index of paper if you think they are directly relevant. SKIP them if you think the given papers are not directly relevant to the question `{{query}}`. 
Before make selection, you need to first generate a thought on why you make such a selection to read the selected papers and skip the rest. 
Consider that the number of papers you can read is limited, you need to make a wise decision on which paper to read. if you feel that one or the papers give to you contains redundant information based on what you already read, you can choose to skip it.
{{inspiration_conversation_history}}

{RESPONSE_FORMAT}
"""

RETRIEVAL_RESPONSE_FORMAT = """
When responding use a markdown code snippet with a JSON object formatted in the \
following schema:
```json
{{   
    "thought": str, //your thought on why you make such a selection or skip them given the research question {{query}}
    "selected_papers": List[str], //list of paper_indexs (str) or string 'skip' (instead of list) if none of them is directly relevant to {{query}}
}}
```
Return only the makrdown and Nothing Else!
"""


READ_PROMPT_PREFIX = """You are an research agent designed to help human perform an literature review tasks from a set of documents.
You are give a task to review information of the given research question: {{query}}.
You are given a paper to read, and you need to summarize the information relevant to the given topic and also the overall information of the document that you feel important or interesting or worth to dive in. 
{{paper_already_read}}
{{findings_so_far}}
The paper you need to read is below and is wrapped between a pair of triple backticks:  
```
{{paper_to_read}}
```
Be careful that the paper you are assigned might not relevant to the research question: {{query}}. But if it is relevant, you also are required to generate a overall thought to reflect your understanding of the given topic `{{query}}`so far, based on this paper, papers you have already read and your findings. 
Be careful about the conflict findings from this paper and the papers you have read (if you have), do not judge which one is more reasonable based on our understanding, just write it to your thought and wait for further exploration!
{{inspiration_conversation_history}}

{RESPONSE_FORMAT}
"""

READ_RESPONSE_FORMAT = """
When responding use a markdown code snippet with a JSON object formatted in the \
following schema:
```json
{{  
    "findings_of_the_paper": str, //the information in the paper that directly related to the given research question `{{query}}`, return `not included` if it does not contain the information that can directly relate to the research question
    "related_to_query": bool, //whether the paper is directly related to the given research question `{{query}}`, 
    "reason_of_exclusion": str, //if your answer was `false` in "related_to_query", you need to provide the reason why you think it is not related to the research question `{{query}}
    "summary_of_the_paper": str, //summary of the paper given to you, include the information that you feel important and worth further exploration
    "thought": str, //your overall understanding of question `{{query}}` so far, not just from this paper but also your previous findings. Do not include irrelevant information here. Pay attention to the length of your thought, it should be no longer than 500 words.    
}}
```
Return only the makrdown and Nothing Else!
"""

#解析包含JSON的markdown字符串
def parse_json_markdown(json_string: str) -> dict:
    try:#尝试引用re和json模块
        re
        json
    except:#引用失败就在except中导入
        import re
        import json
    # Try to find JSON string within triple backticks
    #从三重反引号中提取JSON字符串
    match = re.search(r"```(json)?(.*?)```", json_string, re.DOTALL)
    #若未找到匹配，假设整个字符串都是JSON
    # If no match found, assume the entire string is a JSON string
    if match is None:#如果未找到匹配
        json_str = json_string#假设整个字符串都是JSON字符串
    else:
        #若找到匹配，使用反引用内的内容
        # If match found, use the content within the backticks
        json_str = match.group(2)
    #去除首尾空白和换行
    # Strip whitespace and newlines from the start and end
    json_str = json_str.strip()
    #解析JSON字符串
    # Parse the JSON string into a Python dictionary
    parsed = json.loads(json_str)

    return parsed

#定义反思人类问题指令的动作类
class ReflectOnHuamnQuestionInstruction(BaseAction):
    action_name: str = "reflect"#动作名称
    prompt_prefix: str = REFLECT_ON_HUMAN_QUESTION_INSTRUCTION.format(#初始化提示词（填充响应格式）
        RESPONSE_FORMAT=REFLECT_RESPONSE_FORMAT)
    prompt: PromptTemplate = PromptTemplate(template=prompt_prefix, input_variables=[#创建提示词模板（指定输入变量）
                                            "action_history", "conversation_history", "query"])
    input_variables: List[str] = [#输入变量列表
        "action_history", "conversation_history", "query"]
    #使用重试装饰器：最多重试2次，不等待，如果抛出ValueError异常，则重试，否则返回0，默认值
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(2),#最多重试2次
        wait=tenacity.wait_none(),  # No waiting time between retries重试间无等待
        retry=tenacity.retry_if_exception_type(ValueError),#抛出ValueError异常时重试
        before_sleep=lambda retry_state: print(#输出异常信息
            f"ValueError occurred: {retry_state.outcome.exception()}, retrying..."),
        # Default value when all retries are exhausted所有重试结束后默认返回值
        retry_error_callback=lambda retry_state: 0
    )
    #创建一个异步动作方法
    async def act(self, agent: BaseAgent, query: str, scratch_pad: str):
        log_action("reflect", agent.current_iter)#记录动作日志
        # 初始化OpenAI聊天模型（使用gpt-3.5-turbo-0125,温度0表示确定性输出）
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)
        reflect_chain = LLMChain(llm=llm, prompt=self.prompt)#创建LLM链
        action_history = ""#初始化动作历史
        if len(agent.action_sequence_done) > 0:#如果动作历史列表长度大于0
            action_history = f"Your recent action history are: \n"#添加动作历史
            action_history += "\n".join(#拼接已完成的动作记录
                [f"{action.get('findings', '')}" for action in agent.action_sequence_done])
        #初始化对话记录
        conversation_history = ""
        if len(scratch_pad) > 0:#如果对话记录长度大于0
            #添加对话记录
            conversation_history = f"Conversation history between you and the human expert are: \n"
            conversation_history += "\n".join(#拼接对话记录
                [f"{chat['role']}: {chat['message']}" for chat in scratch_pad])
        #调用LLM生成原始响应
        raw_response = await reflect_chain.apredict(action_history=action_history, conversation_history=conversation_history, query=query)
        parsed_response = parse_json_markdown(raw_response)#响应解析未JSON
        #判断是否包含指令，如果包含则更新智能体启发式对话历史
        if parsed_response["instruction_included"]:
            log_action_res(f"Reflection: {parsed_response['reflection']}")#记录指令
            agent.inspiration_conversation_history = parsed_response["reflection"]#保存指令
        else:#不包含指令
            log_action_res("No specific instruction from human expert")#记录无指令
        #添加到智能体已完成动作列表（时间戳、动作名称、Findings）
        agent.action_sequence_done.append({
            "time_stamp": agent.current_iter,
            "action": "reflect",
            "findings": f"Based on the discussion with human expert, you reflected on your work and thought: `{parsed_response['reflection']}`"
        })

#定义与人类专家进行讨论的指令动作类
class Discuss(BaseAction):
    """"""
    action_name: str = "discuss"#动作名称
    #初始化提示词（填充响应格式）
    prompt_prefix: str = DISCUSS_WITH_HUMAN_PROMPT_PREFIX.format(
        RESPONSE_FORMAT=DISCUSS_RESPONSE_FORMAT)
    #创建提示词模板
    prompt: PromptTemplate = PromptTemplate(template=prompt_prefix, input_variables=[
                                            "paper_already_read", "findings_so_far", "action_history", "conversation_history", "human_expert_question", "query"])
    input_variables: List[str] = ["paper_already_read", "findings_so_far",
                                  "action_history", "conversation_history", "human_expert_question", "query"]#输入变量列表
    #使用重试装饰器：最多重试2次，不等待，如果抛出ValueError异常，则重试，否则返回0，默认值
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(2),#最多重试2次
        wait=tenacity.wait_none(),  # No waiting time between retries重试间没等待时间
        retry=tenacity.retry_if_exception_type(ValueError),#抛出ValueError异常时重试
        before_sleep=lambda retry_state: print(#输出异常信息
            f"ValueError occurred: {retry_state.outcome.exception()}, retrying..."),
        # Default value when all retries are exhausted所有检索结束时默认返回值
        retry_error_callback=lambda retry_state: 0
    )
    #创建一个异步动作方法
    async def act(self, agent: BaseAgent, query: str, scratch_pad: str, working_memory: str, memory: MemoryBank, workspace: Document, command: str):
        log_action("discuss", agent.current_iter)#记录动作日志
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)#初始化OpenAI聊天模型（使用gpt-3.5-turbo-0125,温度0表示确定性输出）
        discuss_chain = LLMChain(llm=llm, prompt=self.prompt)#创建LLM链
        paper_already_read = "You haven't read any paper yet.\n"#初始化已读论文记录
        if len(memory.index_to_docstore_id) > 0:#如果已读论文列表长度大于0
            paper_already_read = f"You have read {len(memory.get_leaf_memories())} papers so far \n"#添加已读论文记录
            paper_already_read += f"And you findings on the most relevant ones are: \n"#添加已读论文的Findings
            paper_already_read += "\n".join([doc.page_content for doc in memory.get_inner_memories()])#添加已读论文的Findings

        findings_so_far = ""#初始化Findings记录
        if len(working_memory):#如果工作记忆长度大于0
            findings_so_far = f"Your working memory so far: \n {working_memory}\n"#添加工作记忆记录

        action_history = ""#初始化动作历史
        if len(agent.action_sequence_done) > 0:#如果动作历史列表长度大于0
            action_history = f"Your recent action history are: \n"#添加动作历史
            # 拼接已完成的动作记录
            action_history += "\n".join(\
                [f"{action['findings']}" for action in agent.action_sequence_done if action.get("findings") is not None])

        conversation_history = ""#初始化对话记录
        if len(scratch_pad) > 0:#如果对话记录长度大于0
            conversation_history = f"Conversation history between you and the human expert are: \n"#添加对话记录
            # 拼接对话记录
            conversation_history += "\n".join(\
                [f"{chat['role']}: {chat['message']}" for chat in scratch_pad])
        # 调用LLM生成原始响应
        if agent.client_handler: #如果有客户端处理句柄
            agent.client_handler.emitAgentWorkingProgress({#发送代理工作进度事件
                "agent_id": agent.agent_id,
                "working_status": "reflecting",
            })
        #调用LLM生成原始响应
        human_expert_question = f"Human: {command}\n Your response: "
        #调用LLM生成原始响应
        raw_response = await discuss_chain.apredict(paper_already_read=paper_already_read, findings_so_far=findings_so_far, action_history=action_history, \
                                                    conversation_history=conversation_history, query=query, human_expert_question=human_expert_question)
        scratch_pad.append({"role": "Human", "message": command})#添加对话记录
        scratch_pad.append({"role": "Agent", "message": raw_response})#添加对话记录
        #添加到智能体已完成动作列表（时间戳、动作名称、Findings）
        agent.action_sequence_done.append({
            "time_stamp": agent.current_iter,
            "action": "dicuss",
            "findings": f"Human expert asked and commented: `{command}` and you responded with: `{raw_response}"
        })
        if agent.client_handler: #如果有客户端处理句柄
            agent.client_handler.emitAgentWorkingProgress({#发送代理工作进度事件
                "agent_id": agent.agent_id,
                "working_status": "reflected",
                "response": raw_response,
            })
        log_action_res(f"Agent: {raw_response}")#记录代理响应
        reflectionAction = ReflectOnHuamnQuestionInstruction()#创建一个反射人类问题的指令
        await reflectionAction.act(agent=agent, query=query, scratch_pad=scratch_pad)#调用反射人类问题的指令
        return raw_response

# 反射人类问题的指令
class RetrieveAction(BaseAction):
    """retrieve action"""
    action_name: str = "retrieve"#动作名称
    # 创建一个LLM链
    prompt_prefix: str = RETRIEVE_PROMPT_PREFIX.format(
        RESPONSE_FORMAT=RETRIEVAL_RESPONSE_FORMAT)
    # 创建一个提示模板
    prompt: PromptTemplate = PromptTemplate(template=prompt_prefix, input_variables=[
                                            "paper_already_read", "findings_so_far", "inspiration_conversation_history", "available_papers", "query"])
    input_variables: List[str] = ["paper_already_read", "findings_so_far",
                                  "inspiration_conversation_history", "available_papers", "query"]
    # 使用重试装饰器
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(2),
        wait=tenacity.wait_none(),  # No waiting time between retries
        retry=tenacity.retry_if_exception_type(ValueError),#捕获异常重试
        #在睡眠等待下次重试前要执行的操作
        before_sleep=lambda retry_state: print(
            f"ValueError occurred: {retry_state.outcome.exception()}, retrying..."),
        # Default value when all retries are exhausted
        #重试耗尽时的默认返回值
        retry_error_callback=lambda retry_state: 0
    )
    # 异步调用LLM进行预测
    async def act(self, agent: BaseAgent, query: str, scratch_pad: str, working_memory: str, memory: MemoryBank, workspace: Any):
        """"""
        log_action("retrieve", agent.current_iter)#记录动作日志
        if len(workspace) == 0:#如果工作空间长度为0
            return
        current_workspace = self.get_iter_workspace(agent, workspace)#获取当前工作空间
        if len(current_workspace) == 0:#如果工作空间长度为0
            log_action_res("No more paper to read")#记录无更多论文可读日志
            ### Emit agent working progress
            agent.client_handler.emitAgentWorkingProgress({#发送代理工作进度事件
                "agent_id": agent.agent_id,
                "working_status": "failed-retrieve-out-of-paper",
            })
            ### end of emit
            return "no more paper to read", []#返回无更多论文可读结果
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)#创建一个ChatOpenAI模型
        retrieve_chain = LLMChain(llm=llm, prompt=self.prompt)#创建一个LLM链

        paper_already_read = "You haven't read any paper yet.\n"#初始化已读论文记录
        if len(memory.index_to_docstore_id) > 0:#如果索引到文档存储ID的长度大于0
            paper_already_read = f"You have read {len(memory.get_leaf_memories())} papers so far \n"#添加已读论文记录

        findings_so_far = ""#初始化 findings_so_far
        if len(working_memory):#如果工作内存长度大于0
            findings_so_far = f"Your findings so far: \n {working_memory}\n"#添加 findings_so_far

        inspiration_conversation_history = ""#初始化inspiration_conversation_history
        if len(inspiration_conversation_history) > 0:#如果inspiration_conversation_history长度大于0
            inspiration_conversation_history += "Based on your previous conversation with human expert, you have the following inspiration: \n"#添加inspiration_conversation_history
            inspiration_conversation_history += agent.inspiration_conversation_history#添加inspiration_conversation_history
            inspiration_conversation_history += "Make sure your action reflect on this inspiration.\n"#添加inspiration_conversation_history
        # 获取当前工作空间
        _available_papers_id_to_doc = {
            str(idx): doc for idx, doc in enumerate(current_workspace)}
        # 为emit添加数据
        _available_papers_id_to_doc_for_emit = {
            str(idx): {
                "metadata": doc.metadata,
                "abstract": doc.metadata.get("AB", "None"),
                "authors": doc.metadata.get("AU", []),
                "id": doc.docstore_id,
            } \
            for idx, doc in enumerate(current_workspace)#添加数据
        }
        # 发送代理工作进度事件
        # Emit agent working progress
        agent.client_handler.emitAgentWorkingProgress({#发送代理工作进度事件
            "agent_id": agent.agent_id,
            "working_status": "start-retrieve",
            "reception_field": list(_available_papers_id_to_doc_for_emit.values())
        })
        # 将可用论文字典中的所有论文内容格式化为带索引的字符串列表，然后用换行符连接成一个完整的字符串
        # 格式为"索引: 论文内容"，便于展示和选择
        available_papers = "\n".join(
            [f"{idx}: {doc.page_content}" for idx, doc in _available_papers_id_to_doc.items()])
        # 调用LLM进行预测
        raw_response = await retrieve_chain.apredict(paper_already_read=paper_already_read, findings_so_far=findings_so_far, inspiration_conversation_history=inspiration_conversation_history, \
                                                     available_papers=available_papers, query=query)
        parsed_response = parse_json_markdown(raw_response)#解析JSON格式的预测结果
        current_thought = parsed_response["thought"]#获取当前thought
        # 如果用户选择跳过，则返回空列表
        if parsed_response["selected_papers"] == "skip":
            selected_papers = []#返回空列表
            log_action_res(f"No paper selected with the reason: {current_thought}")#记录无选择结果日志
            agent.action_sequence_done.append({#添加代理行为序列完成
                "time_stamp": agent.current_iter,
                "action": "finished-retrieve",
                "findings": f"You skipped {len(current_workspace)} papers given to you and thought: `{current_thought}`"
            })
            # 发送代理工作进度事件
            # Emit agent working progress
            if agent.client_handler: 
                agent.client_handler.emitAgentWorkingProgress({#发送代理工作进度事件
                    "agent_id": agent.agent_id,
                    "working_status": "finished-retrieve",
                    "paper_selected": [],  # skip | list of paper index
                    "thought": current_thought
                })
            ###
        else:#如果用户选择 papers
            # 将用户选择的论文内容格式化为带索引的列表，并添加到selected_papers中
            selected_papers = [_available_papers_id_to_doc[str(
                idx)] for idx in parsed_response["selected_papers"]]
            # 将用户选择的论文内容格式化为带索引的列表，并添加到selected_papers_for_emit中
            selected_papers_for_emit = [_available_papers_id_to_doc_for_emit[str(
                idx)] for idx in parsed_response["selected_papers"]]
            # 记录选择结果日志
            log_action_res(f"Selected {len(selected_papers)} papers")
            log_action_res(f"Reason of selection: {current_thought}")
            # 添加代理行为序列完成
            agent.action_sequence_done.append({
                "time_stamp": agent.current_iter,
                "action": "retrieve",
                "findings": f"You selected {len(selected_papers)} papers ({','.join([paper.metadata['TI'] for paper in selected_papers])}) given to you and thought: `{current_thought}`"
            })
            # Emit agent working progress
            if agent.client_handler: 
                agent.client_handler.emitAgentWorkingProgress({
                    "agent_id": agent.agent_id,
                    "working_status": "retrieve",
                    "paper_selected": selected_papers_for_emit,  # skip | list of paper index
                    "thought": current_thought
                })
            ###
        # 更新已访问的论文
        for doc in _available_papers_id_to_doc.values():
            agent.papers_visited.append(doc)
            if doc not in selected_papers:#如果论文没有被选择
                doc.metadata["retrieve_status"] = f"skipped-by-{agent.agent_id}"#更新已访问的论文
            else:#如果论文被选择
                doc.metadata["retrieve_status"] = f"selected-by-{agent.agent_id}"#更新已访问的论文

        return current_thought, selected_papers
    # 获取当前工作空间
    def get_iter_workspace(self, agent: BaseAgent, workspace, capacity: int = 10) -> List[Document]:
        """"""
        candidates = agent.environment_cur.retrieve_candidate_documents_from_workspace(agent)#获取当前工作空间
        return candidates

#定义阅读动作类
class Read(BaseAction):
    """Read and Summarize a single retrieved article"""
    action_name: str = "read"#动作名称
    #初始化提示词（填充响应格式）
    prompt_prefix: str = READ_PROMPT_PREFIX.format(
        RESPONSE_FORMAT=READ_RESPONSE_FORMAT)
    #创建提示模板（指定输入变量）
    prompt: PromptTemplate = PromptTemplate(template=prompt_prefix, input_variables=[
                                            "paper_already_read", "findings_so_far", "inspiration_conversation_history", "paper_to_read", "query"])
    #重试装饰器（同前）
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(2),
        wait=tenacity.wait_none(),  # No waiting time between retries
        retry=tenacity.retry_if_exception_type(ValueError),
        before_sleep=lambda retry_state: print(
            f"ValueError occurred: {retry_state.outcome.exception()}, retrying..."),
        # Default value when all retries are exhausted
        retry_error_callback=lambda retry_state: 0
    )
    #异步执行动作
    async def act(self, agent: BaseAgent, query: str, scratch_pad: str, working_memory: str, memory: MemoryBank, workspace: Document):
        """"""
        log_action("read", agent.current_iter)#记录动作
        if workspace is None:#如果工作空间为空
            return#返回
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)#创建LLMChain，温度设置为0（主要表示确定性输出）
        read_chain = LLMChain(llm=llm, prompt=self.prompt)#创建LLMChain
        paper_already_read = "You haven't read any paper yet.\n"#默认已读论文为空
        if len(memory.index_to_docstore_id) > 0:#如果已读论文数量大于0
            paper_already_read = f"You have read {len(memory.get_leaf_memories())} papers so far: \n"#已读论文
            # paper_already_read += f"And you findings on the most relevant ones are: \n"
            # paper_already_read += "\n".join([doc.page_content for doc in memory.docstore._dict.values()])
        # 获取待读论文内容及用于发送的信息
        paper_to_read = workspace.page_content#待读论文内容
        paper_to_read_for_emit = {#发送的信息
            "metadata": workspace.metadata,
            "abstract": workspace.metadata.get("AB", "None"),
            "authors": workspace.metadata.get("AU", []),
            "id": workspace.docstore_id,
        }
        log_action(f"Reading paper: {workspace.metadata.get('TI')}", agent.current_iter, level=1)#记录阅读论文标题
        #发送开始阅读的工作进度
        ### Emit agent working progress 
        if agent.client_handler: 
            agent.client_handler.emitAgentWorkingProgress({
                "agent_id": agent.agent_id,
                "working_status": "start-read",
                "target_paper": paper_to_read_for_emit,
            })
        ###
        # 获取已读论文
        findings_so_far = ""#默认已读论文为空
        if len(working_memory):#如果已读论文数量大于0
            findings_so_far = f"Your findings so far: \n {working_memory}\n"#已读论文
        # 获取灵感
        inspiration_conversation_history = ""#默认灵感为空
        if len(inspiration_conversation_history) > 0:#如果灵感数量大于0
            inspiration_conversation_history += "Based on your previous conversation with human expert, you have the following inspiration: \n"#添加灵感
            inspiration_conversation_history += agent.inspiration_conversation_history#添加灵感
            inspiration_conversation_history += "Make sure your action reflect on this inspiration.\n"#添加灵感
        # 执行LLMChain
        raw_response = await read_chain.apredict(paper_already_read=paper_already_read, findings_so_far=findings_so_far, inspiration_conversation_history=inspiration_conversation_history, \
                                                 paper_to_read=paper_to_read, query=query)
        parsed_response = parse_json_markdown(raw_response)#解析JSON
        related_to_query_or_not = bool(parsed_response["related_to_query"])#判断论文是否与查询有关
        findings_of_the_paper = parsed_response["findings_of_the_paper"]#获取论文 findings
        summary_of_the_paper = parsed_response["summary_of_the_paper"]#获取论文 summary
        thought = parsed_response["thought"]#获取论文 thoughts
        log_action_res(f"finished reading paper with with decision `{COLOR_INCLUDE if related_to_query_or_not else COLOR_EXCLUDE}` and finding: {findings_of_the_paper}")#记录论文结果
        if related_to_query_or_not:#如果论文与查询有关
            agent.agent_working_memory = thought#更新已读论文
            agent.memory.add_working_summary(#添加已读论文
                texts=[findings_of_the_paper], metadatas=[workspace.metadata])#添加已读论文
            agent.action_sequence_done.append({#添加已读论文
                "time_stamp": agent.current_iter,
                "action": "read",
                "findings": f"You read and consider paper: {workspace.metadata.get('TI')} relevant."
            })
            ### Emit agent working progress 
            if agent.client_handler: #发送工作进度
                agent.client_handler.emitAgentWorkingProgress({#发送工作进度
                    "agent_id": agent.agent_id,
                    "working_status": "finish-read",
                    "target_paper": paper_to_read_for_emit,
                    "decision": "include",
                    "findings": findings_of_the_paper,
                })
            ###
            # log_action_res(f"finished reading paper with finding: {findings_of_the_paper}")
        else:#如果论文与查询无关
            log_action_res(f"reason of exclusion: {parsed_response['reason_of_exclusion']}")#记录排除原因
            agent.action_sequence_done.append({#添加已读论文
                "time_stamp": agent.current_iter,
                "action": "read",
                "findings": f"You read and consider paper: {workspace.metadata.get('TI')} irrelevant, because `{parsed_response['reason_of_exclusion']}`"
            })
            ### Emit agent working progress
            if agent.client_handler: #发送工作进度
                agent.client_handler.emitAgentWorkingProgress({#发送工作进度
                    "agent_id": agent.agent_id,
                    "working_status": "finish-read",
                    "target_paper": paper_to_read_for_emit,
                    "decision": "exclude",
                    "reason_of_exclusion": parsed_response['reason_of_exclusion'],
                })
            ###
        return parsed_response
