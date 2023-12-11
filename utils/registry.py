class Registry:
    def __init__(self,name):
        self.name = name
        self.class_dict = {}

    def register_class(self):
        def _register(cls):
            name = cls.__name__.lower().replace("dataset","")
            self.class_dict[name] = cls
        return _register

    def get(self, name):
        if name not in self.class_dict:
            raise Exception(f"{name} is not registered in {self.name} registry")
        return self.class_dict[name]

    def add(self, cls, name):
        self.class_dict[name] = cls

Datasets = Registry("datasets")
Optimizers = Registry("optimizers")
Losses = Registry("losses")
