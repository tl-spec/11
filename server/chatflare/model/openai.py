import os#用于操作系统相关操作，如环境变量获取
import asyncio # 用于支持异步编程
from openai import OpenAI, AsyncOpenAI# 导入OpenAI的同步和异步客户端

from chatflare.model.base import ModelBase# 从自定义模块导入基础模型类
from dotenv import load_dotenv # 用于加载.env文件中的环境变量
from datetime import datetime # 用于获取当前时间
#定义ChatOpenAI类，继承自ModelBase基础模型类
class ChatOpenAI(ModelBase):
    def __init__(self, model_name="gpt-4o-mini", **kwargs):
        self.model_name = model_name# 模型名称
        if os.getenv('OPENAI_API_KEY') is None:# 如果环境变量OPENAI_API_KEY不存在
            load_dotenv('/app/.env')# 加载.env文件
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))# 创建OpenAI客户端
        self.aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))# 创建异步OpenAI客户端
    #同步预测
    def predict(self, **kwargs):
        """
        """
        #如果kwargs中没有指定model,使用类初始化时model_name
        if 'model' not in kwargs:
            kwargs['model'] = self.model_name

        #调用OpenAI同步客户端的chat.completions.create方法，传入
        response = self.client.chat.completions.create(**kwargs)
        #从响应中获取token使用信息
        # get token usage from response
        usage_info = response.usage
        if usage_info:  #如果存在使用信息
            prompt_tokens = usage_info.prompt_tokens#获取提示词的token使用数量
            completion_tokens = usage_info.completion_tokens#获取生成内容token数量
            total_tokens = usage_info.total_tokens#获取总token使用数量
            
            #打开log-tokens.txt文件，并写入当前时间、模型名称、提示词token使用数量、生成内容token数量、总token使用数量
            with open("log-tokens.txt", "a", encoding="utf-8") as f:
                f.write(
                    f"{datetime.now()}, " 
                    f"[SYNC] Model: {kwargs['model']}, "
                    f"Prompt tokens: {prompt_tokens}, "
                    f"Completion tokens: {completion_tokens}, "
                    f"Total tokens: {total_tokens}\n"
                )

        #根据是否需要返回完整响应决定返回内容
        if 'return_full_response' not in kwargs or kwargs['return_full_response'] == False:
            return response.choices[0].message.content
        else:
            return response

    # 异步预测
    async def apredict(self, **kwargs):
        """
        """
        if 'model' not in kwargs:# 如果没有指定model，使用类初始化时model_name
            kwargs['model'] = self.model_name# 使用类初始化时model_name
        #调用OpenAI异步客户端的chat.completions.create方法，传入参数kwargs
        response = await self.aclient.chat.completions.create(**kwargs)
        #获取token使用信息
        usage_info = response.usage
        if usage_info:# 如果存在使用信息
            print("get usage info")
            prompt_tokens = usage_info.prompt_tokens# 获取提示词的token使用数量
            completion_tokens = usage_info.completion_tokens# 获取生成内容token数量
            total_tokens = usage_info.total_tokens# 获取总token使用数量
            #打开log-tokens.txt文件，并写入当前时间、模型名称、提示词token使用数量、生成内容token数量、总token使用数量
            with open("log-tokens.txt", "a", encoding="utf-8") as f:
                f.write(
                    f"{datetime.now()}, " 
                    f"[ASYNC] Model: {kwargs['model']}, "
                    f"Prompt tokens: {prompt_tokens}, "
                    f"Completion tokens: {completion_tokens}, "
                    f"Total tokens: {total_tokens}\n"
                )
        #根据是否需要返回完整响应决定返回内容
        if 'return_full_response' not in kwargs or kwargs['return_full_response'] == False:
            return response.choices[0].message.content
        else:
            return response


