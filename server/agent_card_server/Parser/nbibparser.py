from agent_card_server.Memory.embeddingdoc import EmbeddingDocument
from agent_card_server.Environment.faiss import FAISSE
from langchain_openai import OpenAIEmbeddings
import uuid
import json 

NBIB_SPLITTER = '- ' 
DEFAULT_JSON_FILE_NAME_PREFIX = "target" 
def parse_nbib_to_dict(file): 
    article_objs = [] 
    with open(file, 'r') as f:
        nbib_raw = f.readlines()  
    article = {} 
    for line in nbib_raw:
        if (line == "\n"):
            for key in article.keys():
                if len(article[key]) == 1:
                    article[key] = article[key][0]
            article_objs.append(article) 
            article = {} 
            continue
        
        if (line[4] == "-"):
            key = line[:4].strip()
            value = line[5:].strip()
            if article.get(key) is not None:
                article[key] = [*article[key], value]
            else:
                article[key] = [value]
        elif (line[:4] == "    "):
            value = article[key].pop()
            if len(article[key]) > 0:
                article[key] = [*article[key], value + " " + line[5:].strip()]
            else:
                article[key] = [value + " " + line[5:].strip()]
        else:
            print("unregconized line: ", line)  
    return article_objs


def parse_nbib_to_json(file):
    article_objs = parse_nbib_to_dict(file)
    json_file_name = get_prefix(file) + ".json" 
    print(json_file_name)
    with open(json_file_name, 'w') as f:
        json.dump(article_objs, f)


# write a function to get the prefix of a file:
def get_prefix(file):  
    # use regular expression to extract file name and its path remove the suffix
    import re 
    pattern = re.compile(r'(.*)\.nbib')
    match = pattern.match(file)
    if match:
        return match.group(1)
    else: 
        return f"./{DEFAULT_JSON_FILE_NAME_PREFIX}"

def read_json(file):
    with open(file, 'r') as f:
        return json.load(f)

def raw_docs_to_document(docs): 
    res = [] 
    ids = [str(uuid.uuid4()) for _ in docs]
    for i, doc in enumerate(docs): 
        if doc.get("AB") is None: 
            continue
        doc["source"] = doc.get("AB") + " " + doc.get("TI") 
        res.append(EmbeddingDocument(embedding=None, docstore_id = ids[i], page_content = doc.get("TI") + "\n" + doc.get("AB"), metadata = doc))
    return res

    
def raw_docs_to_faiss_docstore(docs):#将原始文档转换为FAISS向量存储
    res = [] 
    texts = [] 
    metadatas = [] 
    for i, doc in enumerate(docs): 
        if doc.get("AB") is None: #如果没有AB，则跳过
            continue
        texts.append(doc.get("AB") + " " + doc.get("TI"))#构建文本（文本组合为摘要+标题）
        metadatas.append(doc)
        
    # ids = [str(uuid.uuid4()) for _ in texts]    
    return FAISSE.from_texts(
        texts = texts,#包含文档摘要和标题组合的文本列表
        embedding = OpenAIEmbeddings(),#使用OpenAI的嵌入模型将文本转换为向量
        metadatas = metadatas,#存储原始文档的元数据信息
    )#最终返回一个FAISS向量存储对象，用于后续的相似性检索

