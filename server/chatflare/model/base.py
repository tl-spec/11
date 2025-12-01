from abc import ABC, abstractmethod 
## 从abc模块导入ABC（抽象基类）和abstractmethod（抽象方法装饰器）
# 定义一个名为ModelBase的抽象基类，继承自ABC
class ModelBase(ABC):
    # 定义类的初始化方法，接收任意关键字参数（**kwargs）
    def __init__(self, **kwargs):
        pass#初始化方法体为空，暂不执行任何操作
    # 定义类的__repr__方法，用于返回对象的字符串表示
    def __repr__(self):
        """
        Returns a string representation of the model.
        """#返回对象的字符串表示，内容为当前对象的model_name属性值，若不存在则为‘unkown_model’
        return f"{getattr(self, 'model_name', 'unknown_model')}"
    #用abstractmethod装饰器定义抽象方法predict，强制子类必须实现该方法
    @abstractmethod 
    def predict(self, **kwargs):
        pass
    #用abstractmethod装饰器定义抽象方法apredict，强制子类必须实现该方法
    @abstractmethod
    def apredict(self, **kwargs):
        pass #抽象方法体为空，仅作为接口定义
        