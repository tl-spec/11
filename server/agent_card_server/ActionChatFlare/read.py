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

# from chatflare.model.qwen import QwenLLM

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
{{inclusion_criteria}}

Be careful about the conflict findings from this paper and the papers you have read (if you have), do not judge which one is more reasonable based on our understanding, just write it to your thought and wait for further exploration!
{{inspiration_conversation_history}}

{RESPONSE_FORMAT}
"""


READ_RESPONSE_FORMAT = """
When responding use a JSON object formatted in the \
following schema:
```json
{{  
    "analysis": str, //your analysis of the paper, including the information in the paper that directly related to the given research question `{{query}}` and the information that you feel important and worth further exploration. 
    "response_preparation_analysis": str, //Consider user's requirement from the conversation history {{inspiration_conversation_history}}  chrish this, and analyze how should you when generating response.
    "related_to_query": bool, //whether the paper is directly related to the given research question `{query}`, 
    "reason_of_exclusion": str, //if your answer was `false` in "related_to_query", you need to provide the reason why you think it is not related to the research question `{{query}}
    "summary_of_the_paper": str, //if no specific instructions from human, summary of the paper given to you, including the information in the paper that directly related to the given research question `{{query}}` and the information that you feel important and worth further exploration, return `not included` if it does not contain the information that can directly relate to the research question.
    "summary_phrase": str, //if no instructions from human, return a short phrase that summarize the paper, it should be no longer than 3 words to capture the term or findings of this paper relate to the research question. e.g. if the research query is about "side effect of MPH", and the paper is about a specific type of side effect "insominia", the summary phrase could be "insominia"
    "thought": str, //your overall understanding of question `{query}` so far, not just from this paper but also your previous findings. Do not include irrelevant information here. Pay attention to the length of your thought, it should be no longer than 500 words.    
}}
```
Human expert's instruction and guidance from the conversation history is the most important information you should consider when making decision, generating summary and key phrases, cherish them first 
"""

#创建LLM链
def create_read_chain(model_name='gpt-4o-mini'):
    prompt_prefix = READ_PROMPT_PREFIX.\
        format(RESPONSE_FORMAT=READ_RESPONSE_FORMAT)#创建提示模板
    prompt = PromptTemplate(template=prompt_prefix)#将提示模板封装成了标准化的prompttemplate对象
    if "llama" in model_name:#如果是llama模型
        model = LlamaBedrock(model_name=model_name)
    elif 'gpt' in model_name:#如果是gpt模型
        model = ChatOpenAI(model_name=model_name)
    elif 'qwen' in model_name:
        model=QwenLLM(model_name=model_name,
                      api_key=os.getenv("QWEN_API_KEY"),
                      base_url=os.getenv("QWEN_BASE_URL"))
    else: #其他情况默认使用ChatOpenAI模型
        model = ChatOpenAI(model_name=model_name)
    chain = BaseChain(model, prompt, json_mode=True)#创建并返回基础链（包括模型、提示和JSON模型）
    return chain

#创建阅读动作类，继承自基础动作类
class ReadAction(BaseAction):
    def __init__(self, action_name="read", model_name: str=None):
        #根据是否提供模型名称，创建对应的运行链
        if model_name is None: 
            runnable = create_read_chain()
        else: 
            runnable = create_read_chain(model_name=model_name)
        #调用父类构造函数初始化动作
        super().__init__(action_name, runnable)
    #切换运行模型的方法
    def switch_runnable_model(self, model_name: str):
        self.runnable = create_read_chain(model_name=model_name)
    #异步执行方法
    async def arun(self, thread: GraphTraverseThread):
        #获取研究问题
        query = thread.task.research_query#获取研究问题
        paper_already_read = "You haven't read any paper yet"#初始化已读论文信息
        if len(thread.graph_state.paper_visited_in_commits) > 0:#如果有已读论文
            documents_visited = set()#创建已读论文集合
            for commit, docs in thread.graph_state.paper_visited_in_commits.items():#遍历已读论文
                documents_visited.update(docs)#添加已读论文
            paper_already_read = f"You have ready {len(documents_visited)} papers so far\n"#·创建已读论文信息

        findings_so_far = ""#初始化迄今为止的Findings信息
        if thread.graph_state.memory.working_memory and len(thread.graph_state.memory.working_memory) > 0:#如果有工作记忆且不为空，更新发现信息
            findings_so_far = f"Your findings so far from previous actions: {thread.graph_state.memory.working_memory}"

        inspiration_conversation_history = ""#初始化启发式对话历史
        #如果有启发式对话历史且不为空，更新启发式对话信息
        if thread.graph_state.memory.inspiration_conversation_history and len(thread.graph_state.memory.inspiration_conversation_history) > 0:
            inspiration_conversation_history = f"Based on your previous conversation with human expert, you have the following inspiration: \n{thread.graph_state.memory.inspiration_conversation_history} Make sure your action reflect on this inspiration.\n"
            inspiration_conversation_history += "Human expert's instruction and guidance from the conversation history is the most important information you should consider when making decision, generating summary and key phrases."
        print("inspiration_conversation_history")
        print(inspiration_conversation_history)
        #从待阅读论文列表中取出一篇论文
        paper_to_read_obj = thread.graph_state.cached_work['papers_to_read'].pop()
        #获取论文内容
        paper_to_read = paper_to_read_obj.page_content

        inclusion_criteria = ""#初始化纳入标准
        if thread.task.in_criteria:#如果有纳入标准，更新该信息
            inclusion_criteria = f"Inclusion criteria: {thread.task.in_criteria}"
        #记录正在阅读的论文信息
        log_action(f"Reading paper: {paper_to_read_obj.metadata.get('TI')}, id: {paper_to_read_obj.docstore_id}", len(
            thread.branch.commits)-1, level=1)
        #准备用于发射的论文信息（元数据、摘要、作者、id）
        paper_to_read_for_emit = {
            "metadata": paper_to_read_obj.metadata,
            "abstract": paper_to_read_obj.metadata.get("AB", "None"),
            "authors": paper_to_read_obj.metadata.get("AU", []),
            "id": paper_to_read_obj.docstore_id,
        }
        #更新文章处理历史为reading状态
        tmp_article_processing_history = thread.article_processing_history 
        tmp_article_processing_history[paper_to_read_obj.docstore_id] = "reading"
        #向客户端发射代理工作进度（开始阅读）
        thread.client_handler.emitAgentWorkingProgress({
            "agent_id": thread.agent.agent_id,
            "working_status": "start-read",
            "target_paper": paper_to_read_for_emit,
            "branch": thread.branch.id,#分支id
            "read_path": self.get_read_path(thread.branch.commits, thread) + [paper_to_read_for_emit], #阅读路径
            "article_processing_history": tmp_article_processing_history#文章处理历史
        })
        #异步调用LLM链获取输出
        output = await self.runnable.apredict(query=query, paper_already_read=paper_already_read, findings_so_far=findings_so_far, inspiration_conversation_history=inspiration_conversation_history, \
                                              paper_to_read=paper_to_read, inclusion_criteria=inclusion_criteria)
        self.sync_with_thread(output, paper_to_read_obj, thread)#将输出与线程同步

        return output

    def run(self):#同步方法
        query = thread.task.research_query#获取研究问题
        paper_already_read = "You haven't read any paper yet"#初始化已读论文信息
        if len(thread.graph_state.paper_visited_in_commits) > 0:#如果有已读论文
            documents_visited = set()#创建已读论文集合
            for commit, docs in thread.graph_state.paper_visited_in_commits.items():#遍历已读论文
                documents_visited.update(docs)#添加已读论文
            paper_already_read = f"You have ready {len(documents_visited)} papers so far\n"#创建已读论文信息

        findings_so_far = ""#初始化迄今为止的Findings信息
        if thread.graph_state.memory.working_memory and len(thread.graph_state.memory.working_memory) > 0:#如果有工作记忆且不为空，更新发现信息
            findings_so_far = f"Your findings so far: {thread.graph_state.memory.working_memory}"#更新发现信息

        inspiration_conversation_history = ""#初始化启发式对话历史
        if thread.graph_state.memory.inspiration_conversation_history and len(thread.graph_state.memory.inspiration_conversation_history) > 0:
            #如果启发式对话历史且不为空，更新启发式对话信息
            inspiration_conversation_history = (f"Based on your previous conversation with human expert, you have the following inspiration: \n{thread.graph_state.memory.inspiration_conversation_history} "
                                                f"Make sure your action reflect on this inspiration.\n")

        inclusion_criteria = ""#初始化纳入标准
        if thread.task.in_criteria:#如果有纳入标准，更新该信息
            inclusion_criteria = f"Inclusion criteria: {thread.task.in_criteria}"#更新纳入标准
        #从待阅读论文列表中取出一篇论文
        paper_to_read_obj = thread.graph_state.cached_work['papers_to_read'].pop()
        paper_to_read = paper_to_read_obj.page_content#获取论文内容
        #异步调用LLM链获取输出
        output = self.runnable.apredict(query=query, paper_already_read=paper_already_read, findings_so_far=findings_so_far,
                                        inspiration_conversation_history=inspiration_conversation_history, paper_to_read=paper_to_read, inclusion_criteria=inclusion_criteria)
        #将输出与线程同步
        self.sync_with_thread(output, paper_to_read_obj, thread)
        #返回输出
        if self.runnable.JSON_MODE:
            return json.loads(output)
        return output
#输出与线程同步的方法
    def sync_with_thread(self, output, paper_to_read_obj, thread):
        #获取论文是否与查询相关的布尔值
        related_to_query_or_not = bool(output["related_to_query"])
        #获取论文摘要
        # findings_of_the_paper = output["findings_of_the_paper"]
        summary_of_the_paper = output["summary_of_the_paper"]
        keywords = output["summary_phrase"]#获取关键词
        thought = output["thought"]#获取思考（整体思考）
        analysis = output["analysis"]#获取分析内容
        response_preparation_analysis = output["response_preparation_analysis"]#获取准备响应的分析
        #记录阅读完成的结果日志
        log_action_res(
            f"finished reading paper with with decision `{COLOR_INCLUDE if related_to_query_or_not else COLOR_EXCLUDE}` and finding: {summary_of_the_paper}")
        print(response_preparation_analysis)
        #准备用于发射的论文信息（元数据、摘要、作者、id）
        paper_to_read_for_emit = {
            "metadata": paper_to_read_obj.metadata,
            "abstract": paper_to_read_obj.metadata.get("AB", "None"),
            "authors": paper_to_read_obj.metadata.get("AU", []),
            "id": paper_to_read_obj.docstore_id,
        }
        #准备用于发射的阅读结果对象
        read_result_obj = {
            "agentId": thread.agent.agent_id,#智能体id
            "output": output,#输出
            "content": summary_of_the_paper if related_to_query_or_not else "exclude",# 内容
            "meta": {
                "paper_id": paper_to_read_obj.docstore_id,#论文id
                "thought": thought,#思考
                "TI": paper_to_read_obj.metadata.get("TI"),    # 标题
                "summary_keywords": keywords,#关键词
            },
            "update_longterm_memory": related_to_query_or_not,#是否更新长期记忆
            "agent_current_position": paper_to_read_obj.docstore_id, #当前位置
            "cached_work": copy.deepcopy(thread.graph_state.cached_work)#缓存工作
        }
        #更新智能体当前位置为该论文ID
        thread.graph_state.agent_current_position = paper_to_read_obj.docstore_id
        read_result_blob = Blob("read", read_result_obj)#创建一个名为read的blob对象
        commit = Commit.from_blobs([read_result_blob])#创建一个名为commit的Commit对象
        thread.add_commit(commit)#添加Commit对象到线程中
        #处理用户手动拖动阅读文章的情况
        ## **Important! This is to handle the case when user mannually drag to read artifcle
        if paper_to_read_obj.docstore_id not in thread.graph_state.visited_paperids: #如果该论文ID不在已访问的论文ID列表中
            thread.graph_state.paper_visited_in_commits[commit.id] = [paper_to_read_obj.docstore_id]#添加该论文ID到已访问的论文ID列表中

        # 如果该论文不与查询相关，则记录排除原因
        if not related_to_query_or_not:
            log_action_res(
                f"reason of exclusion: {output['reason_of_exclusion']}")#记录排除原因
            #发送排除信息
            thread.client_handler.emitAgentWorkingProgress({
                "agent_id": thread.agent.agent_id,#智能体id
                "working_status": "finish-read",#工作状态
                "target_paper": paper_to_read_for_emit,#目标论文
                "decision": "exclude",#决策
                "reason_of_exclusion": output['reason_of_exclusion'],#排除原因
                "branch": thread.branch.id,# # 标识当前操作属于哪个分支
                "read_path": self.get_read_path(thread.branch.commits, thread), #读取路径
                "article_processing_history": thread.article_processing_history #文章处理历史
            })

        else:#如果该论文与查询相关
            thread.graph_state.cached_work["papers_to_synthesize"].append(read_result_blob.id)#添加该论文ID到缓存工作列表中
            thread.client_handler.emitAgentWorkingProgress({#发送包含信息
                "agent_id": thread.agent.agent_id,#智能体id
                "working_status": "finish-read",#工作状态
                "target_paper": paper_to_read_for_emit,#目标论文
                "decision": "include",#决策
                "findings": thought,#思考
                "hierarchy": thread.graph_state.memory.output_memories_hierarchy(),#获取线程图状态内存中的输出记忆层级结构
                "branch": thread.branch.id,# # 标识当前操作属于哪个分支
                "read_path": self.get_read_path(thread.branch.commits, thread), #读取路径
                "article_processing_history": thread.article_processing_history #文章处理历史
            })
            
#获取阅读路径的方法
    def get_read_path(self, commits, thread):
        read_path = []#创建一个名为read_path的列表
        for commit in commits:#遍历所有提交
            if commit.blobs[0].task_type == "read" and commit.blobs[0].result.get("meta") and commit.blobs[0].result["meta"].get("paper_id"):#如果该提交包含阅读任务并且包含论文ID
                paperId = commit.blobs[0].result["meta"]["paper_id"]#获取论文ID
                if paperId and len(paperId) > 0:#如果论文ID有效
                    paperObj = thread.graph_state.environment.literature_bank.docstore._dict[paperId]#获取论文对象
                    read_path.append({#添加一个字典到read_path中
                        "metadata": paperObj.metadata,
                        "abstract": paperObj.metadata.get("AB", "None"),
                        "authors": paperObj.metadata.get("AU", []),
                        "id": paperObj.docstore_id,
                    })
        return read_path
