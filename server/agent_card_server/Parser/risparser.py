from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain.docstore.document import Document
from langchain_community.vectorstores.faiss import FAISS
from agent_card_server.Environment.environment import InformationSeekingEnvironment, FAISSE
from agent_card_server.Parser.nbibparser import raw_docs_to_faiss_docstore
from agent_card_server.connection.firebase_handler import save_project

import uuid
from datetime import datetime


def parse_ris_file(ris_filename):
    references = []  # 存放所有参考文献信息的列表
    reference = {}  # 存放每条参考文献的信息的字典
    current_tag = None  # 追踪正在处理的标签
    with open(ris_filename, 'r', encoding='utf-8') as file:
        for line in file:#逐行读取文件
            line = line.strip()#去除当前读取行的首尾空白字符（如空格、换行符等）
            # 检测空行作为参考文献的条目
            if not line:
                if reference:  # 如果当前参考文献为非空，将其添加到references列表中
                    references.append(reference)
                    reference = {}  # 重置reference
                current_tag = None  # 重置当前标签
                continue
            # 检查行是否包含标签和值 (RIS 格式 "XX  - Value")
            if len(line) >= 6 and line[2:6] == '  - ':
                current_tag = line[:2]  # 提取前两个标签 (前两个字符)
                data = line[6:]  # 提取Value(在破折号和空格之后)
                # 处理可能出现多次的标签 (like 作者 FAU, 关键字 KW, etc.)
                if current_tag in reference:#如果当前标签已经存在
                    if isinstance(reference[current_tag], list):#如果当前标签已存在，判断该值是否是为列表
                        reference[current_tag].append(data)#如果是，就将新数据追加到列表中（列表样式）
                    else:#否则
                        reference[current_tag] = [reference[current_tag], data]#将原值与新值组成列表储存列表储存（字典样式）
                else:#如果是新标签，就正常存储数据
                    reference[current_tag] = data
            else:
                # 处理持续行: 如果没有找到新的标签，将持续行加入上一个标签的最新加入的value中
                if current_tag and current_tag in reference:
                    if isinstance(reference[current_tag], list):#如果该标签已存在，判断该值是否为列表
                        reference[current_tag][-1] += ' ' + line#如果是，将新数据追加到列表最后一个元素中（列表样式）
                    else:
                        reference[current_tag] += ' ' + line#否则，直接将数据加入该标签字符串值中（字典样式）
    if reference:# 如果文件没有空行结束，添加一个新的参考文献
        references.append(reference)
    title_chache = {} #缓存标题
    deduped_references = []#去重
    for doc in references: #遍历所有参考文献
        title = doc.get('TI') #根据标题（TI字段）进行去重，保留首次出现的文献
        if not title: #判断标题是否为空
            deduped_references.append(doc)#如果标题为空，直接保留该文献，因为无法通过标题去重
            continue 
        else:#标题不为空
            if title not in title_chache:#如果标题没有被缓存过
                title_chache[title] = True #现在缓存标题
                deduped_references.append(doc)#直接保留这条文献
    return deduped_references

def getProjectTemplate(projectId, projectName, numOfArticles, date):
    return {
        "projectId": projectId, #唯一项目ID（UUID生成）
        "projectName": projectName,#项目名称（从文件名提取）
        "numOfArticles": numOfArticles, #文献数量
        "date": date,#处理时间
        "path": f"projects/{projectId}"#云端存储路径
    }


def process_parsed_doc(references, project_id, project_name, date): 
    target_documents = raw_docs_to_faiss_docstore(references)
    target_documents.save_local(f"agent_card_server/data/{project_id}")
    return f"agent_card_server/data/{project_id}"
    
def process_ris_pipeline(ris_filename, project_name): 
    ## extract project name as the last part after '/' before '.ris'
    project_id = str(uuid.uuid4())#生成唯一项目ID和当前时间戳，唯一的项目ID，确保项目不重复
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    references = parse_ris_file(ris_filename)#解析RIS文件为参考文献列表
    #处理并保存文档
    local_path = process_parsed_doc(references, project_id, project_name, date)
    #构造项目信息
    projectDB = getProjectTemplate(project_id, project_name, len(references), date)
    print("check projectDB")
    print(projectDB)
    #保存项目数据，并返回项目ID
    save_project(project_id, projectDB, local_path)
    return project_id
