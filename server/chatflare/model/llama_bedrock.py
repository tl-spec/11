import os
import asyncio
from openai import OpenAI, AsyncOpenAI

from chatflare.model.base import ModelBase
from dotenv import load_dotenv
from datetime import datetime

import boto3 
import aioboto3
import json



def extract_json_content(text):#从文本中提取被json和包裹的JSON内容
    """
    Extract the content inside a JSON blob surrounded by ```json and ```.
    
    Args:
        text (str): The input string containing the JSON blob.
    包含JSON blob的输入字符串。
    Returns:
        str: The extracted JSON content, or None if not found.
        提取的JSON内容，如果没有找到则为None
    """
    import re
    # Define the regex pattern to extract content between ```json and ```
    #定义regex模式来提取json和json之间的内容。
    pattern = r"```json\n(.*?)```"
    
    # Use re.DOTALL to match across multiple lines
    #使用re.DOTALL跨多行进行匹配
    match = re.search(pattern, text, re.DOTALL)
    
    # Return the matched group or None if no match is found
    #返回匹配的组，如果没有找到匹配则返回None
    return match.group(1) if match else None


class LlamaBedrock:
    def __init__(self, model_name="meta.llama3-3-70b-instruct-v1:0", **kwargs):
        self.model_name = model_name#默认使用llama3-3-70b-instruct-v1:0模型
        if os.getenv('AWS_ACCESS_KEY_ID') is None:#如果没有设置AWS_ACCESS_KEY_ID环境变量
            ## check if there is an /app/bedrock.env file
            if os.path.exists('/app/bedrock.env'):#如果存在/app/bedrock.env文件
                load_dotenv('/app/bedrock.env')#加载bedrock.env文件
            else:
                load_dotenv('bedrock.env')#如果不存在bedrock.env文件，则加载.env文件
    #同步调用模型的方法
    def predict(self, **kwargs):
        if 'modelId' not in kwargs:#如果kwargs中没有modelId字段，则使用类初始化时的模型名称
            kwargs['modelId'] = self.model_name#设置模型名称，默认使用llama3-3-70b-instruct-v1:0
        kwargs["contentType"] = "application/json" #设置内容类型为JSON
        kwargs["accept"] = "application/json" #设置接受的响应类型为JSON
        #创建bedrock-runtime的同步客户端，指定区域为us-east-2
        client = boto3.client("bedrock-runtime", region_name="us-east-2")
        #调用模型，传入参数
        response = client.invoke_model(**kwargs)
        #读取并解码响应内容
        response_content = response['body'].read().decode('utf-8')
        response_obj = json.loads(response_content)#将JSON字符串解析为Python对象
        #检查响应中是否包含令牌计数信息
        if "prompt_token_count" in response_obj and "generation_token_count" in response_obj:
            prompt_tokens = response_obj["prompt_token_count"]#获取提示令牌数
            completion_tokens = response_obj["generation_token_count"]#获取生成令牌数
            total_tokens = prompt_tokens + completion_tokens#计算总令牌数
            with open("log-tokens.txt", "a", encoding="utf-8") as f:#将令牌使用情况追加写入log-tokens.txt文件
                f.write(
                    f"{datetime.now()}, " #当前时间
                    f"[ASYNC] Model: {kwargs['modelId']}, "#模型ID（此处ASYNC可能是笔误，应为SYNC）
                    f"Prompt tokens: {prompt_tokens}, "#提示词令牌数
                    f"Completion tokens: {completion_tokens}, "#生成内容令牌数
                    f"Total tokens: {total_tokens}\n"#总令牌数
                )
        #从模型生成的内容中提取JSON内容
        json_blob = extract_json_content(response_obj['generation'])
        return json_blob#返回提取的JSON内容
    #异步调用模型的方法
    async def apredict(self, **kwargs):
        if 'modelId' not in kwargs:#如果kwargs中没有modelId字段，则使用类初始化时的模型名称
            kwargs['modelId'] = self.model_name
        kwargs["contentType"] = "application/json" #设置内容类型为JSON
        kwargs["accept"] = "application/json" #设置接受的响应类型为JSON
        session = aioboto3.Session()#创建一个aioboto3会话
        async with session.client("bedrock-runtime", region_name="us-east-2") as client:#创建一个异步客户端
            response = await client.invoke_model(**kwargs)#调用模型，传入参数
            response_content = await response['body'].read()#读取并解码响应内容
            response_obj = json.loads(response_content)#将JSON字符串解析为Python对象
            if "prompt_token_count" in response_obj and "generation_token_count" in response_obj:#检查响应中是否包含令牌计数信息
                prompt_tokens = response_obj["prompt_token_count"]#获取提示令牌数
                completion_tokens = response_obj["generation_token_count"]#获取生成内容令牌数
                total_tokens = prompt_tokens + completion_tokens#总令牌数
                with open("log-tokens.txt", "a", encoding="utf-8") as f:#将令牌使用情况追加写入log-tokens.txt文件
                    f.write(
                        f"{datetime.now()}, " #当前时间
                        f"[ASYNC] Model: {kwargs['modelId']}, "#模型ID（异步调用标记）
                        f"Prompt tokens: {prompt_tokens}, "#提示词令牌数
                        f"Completion tokens: {completion_tokens}, "#生成内容令牌数字
                        f"Total tokens: {total_tokens}\n"#总令牌数
                    )
            print(response_obj)#打印响应对象
            json_blob = extract_json_content(response_obj['generation']) or response_obj['generation']#从模型生成的内容中提取JSON内容
            print(json_blob)
            return json_blob
        return None