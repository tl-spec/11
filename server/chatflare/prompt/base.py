import string


class PromptTemplate:
    """
    A class to represent a prompt template.

    Example usage: 
        .. code-block:: python
            from chatflare.prompt.base import PromptTemplate 
            template = "Hello {name}, how are you?"
            prompt = PromptTemplate(template)
            print(prompt.render(name="John"))
            # Output: Hello John, how are you?
    """

    def __init__(self, template, **kwargs):
        self.template = template #传入字符串模板保存为实例属性
        self._variables = self._get_variables()#获取模板中的变量

    def __repr__(self):#描述对象
        return f"PromptTemplate(variables={self.variables}, template={self.template[:1000] + '...' if len(self.template) > 1000 else self.template})"
    # 返回格式为 PromptTemplate(variables=变量列表, template=模板内容)：
    # 若模板长度超过 1000 字符，模板内容只显示前 1000 字符并加 ...；否则显示完整模板。

    def _get_variables(self):
        if self.template:#模板不为空
            formatter = string.Formatter()#创建格式化对象
            return [field for _, field, _, _ in formatter.parse(self.template) if field] #获取模板中的变量

    @property
    def variables(self):
        return self._variables

    def render(self, **kwargs):#渲染模板
        return self.template.format(**kwargs)#调用字符串的format方法，用kwargs中的值替换模板中的变量
    # 将模板中的变量替换为对应的值，并返回替换后的模板
    def render_llama(self, **kwargs):
        prompt = self.template.format(**kwargs)#填充模板变量，得到基础提示词prompt
        #将prompt包裹在llama模型要求的特殊标记中
        # （如<|begin_of_text|><|start_header_id|>等），构建最终格式的字符串并返回
        formatted_prompt = f"""
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>
        {prompt}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """
        return formatted_prompt
    #自动填充方法
    def render_autofill(self, **kwargs):
        for variable in self._variables:#遍历_variables中所有变量
            if variable not in kwargs:#检查kwargs中是否包含该变量
                kwargs[variable] = ''#如果没有，则将变量的值设置为空字符串
        return self.template.format(**kwargs)#返回填充后的模板


