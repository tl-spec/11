# 导入所需的基础类和模块
from chatflare.model.base import ModelBase # 导入模型基类，用于与LLM模型交互
from chatflare.prompt.base import PromptTemplate # 导入提示模板基类，用于处理提示词模板
# import tenacity  # 注释掉的重试库，可能用于实现重试逻辑
import json  # 导入json模块，用于JSON格式处理

class BaseChain:
    """基础链类，用于连接提示模板和模型，处理输入输出逻辑"""
    def __init__(self, model:ModelBase, prompt_template:PromptTemplate, **kwargs):
        self.model = model   # 保存模型实例
        self.prompt_template = prompt_template  # 保存提示模板实例
        # 初始化JSON模式，默认为False，若kwargs中有JSON_MODE则使用其值
        self.JSON_MODE = False if 'JSON_MODE' not in kwargs else kwargs['JSON_MODE']
        # 兼容旧参数名json_mode，若为True则开启JSON模式
        if 'json_mode' in kwargs and kwargs['json_mode']:
            self.turn_on_json_mode()
    
    def __repr__(self):
        """返回对象的字符串表示，便于调试和打印"""
        return f"BaseChain(model={self.model}, json_mode={self.JSON_MODE}, prompt_template={self.prompt_template.variables})"

    @property
    def variables(self):
        """属性方法，返回提示模板中需要的变量列表"""
        if self.prompt_template:
            return self.prompt_template.variables#获取提示模板中需要的变量列表
        return []#返回空列表
    
    def turn_on_json_mode(self):
        """开启JSON模式，用于返回JSON格式的输出"""
        self.JSON_MODE = True

    def predict(self, **kwargs):
        prompt_variables = self.prompt_template.variables#获取提示模板中需要的变量列表
        #筛选出提示模板中需要的变量
        prompt_kwargs = {k:v for k,v in kwargs.items() if k in prompt_variables}
        #筛选出模型中需要的参数
        outer_kwargs = {k:v for k,v in kwargs.items() if k not in prompt_variables}
        if 'llama' in self.model.model_name: #如果是llama模型
            rendered_prompt = self.prompt_template.render_llama(**prompt_kwargs)#使用llama的提示模板渲染
            return self.model.predict(
                body=json.dumps({
                    'prompt': rendered_prompt, #提示词
                    'max_gen_len': 4096, #最大生成长度
                    'temperature': 0.1, #温度参数，控制输出随机性
                    'top_p': 0.9, #核采样参数，控制输出多样性
                }).encode('utf-8'),#转换为JSON字符串并编码为字节
            )
        else:#如果不是llama模型
            if self.JSON_MODE:#如果是JSON模式
                outer_kwargs['response_format'] = {"type": "json_object"}#设置响应格式为JSON对象
            rendered_prompt = self.prompt_template.render(**prompt_kwargs)#使用提示模板渲染
            return self.model.predict(
                messages=[
                    {
                        "role": "user",#用户角色
                        "content": rendered_prompt#提示词
                    }
                ],
                model=self.model.model_name,#模型名称
                temperature=0.2, #温度参数，控制输出随机性
                **outer_kwargs#其他参数
            )

    # @tenacity.retry(
    #     stop=tenacity.stop_after_attempt(2),
    #     wait=tenacity.wait_none(),  # No waiting time between retries
    #     retry=tenacity.retry_if_exception_type(ValueError),
    #     before_sleep=lambda retry_state: print(
    #         f"ValueError occurred: {retry_state.outcome.exception()}, retrying..."),
    #     # Default value when all retries are exhausted
    #     retry_error_callback=lambda retry_state: 0
    # )
    #异步调用模型进行预测
    async def apredict(self, **kwargs):
        prompt_variables = self.prompt_template.variables#获取提示模板中需要的变量列表
        prompt_kwargs = {k:v for k,v in kwargs.items() if k in prompt_variables}#筛选出提示模板中需要的变量
        outer_kwargs = {k:v for k,v in kwargs.items() if k not in prompt_variables and k != "debug"}#筛选出模型中需要的参数
        if 'llama' in self.model.model_name:#如果是llama模型
            rendered_prompt = self.prompt_template.render_llama(**prompt_kwargs)
            # 调用模型进行预测
            output = await self.model.apredict(
                body=json.dumps({
                    'prompt': rendered_prompt, #提示词
                    'max_gen_len': 8192, #最大生成长度
                    'temperature': 0.1, #温度参数，控制输出随机性
                    'top_p': 0.9, #核采样参数，控制输出多样性
                }).encode('utf-8'),#转换为JSON字符串并编码为字节
            )
            if self.JSON_MODE:#如果是JSON模式
                return json.loads(output)#解析JSON字符串
            return output
            
        else: #如果不是llama模型
            if self.JSON_MODE:#如果是JSON模式
                outer_kwargs['response_format'] = {"type": "json_object"}#设置响应格式为JSON对象
            rendered_prompt = self.prompt_template.render(**prompt_kwargs)#使用提示模板渲染
            output = await self.model.apredict(
                messages=[
                    {
                        "role": "user",#用户角色
                        "content": rendered_prompt#提示词
                    }
                ],
                model=self.model.model_name,#模型名称
                temperature=0, #温度参数，控制输出随机性
                **outer_kwargs#其他参数
            )
            if "debug" in kwargs and kwargs["debug"]:#如果debug参数为True
                print(f"logging: apredict output: {output}")#打印日志
            
            if self.JSON_MODE:#如果是JSON模式
                return json.loads(output)#解析JSON字符串
            return output


    