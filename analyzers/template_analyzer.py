from .container import DataContainer

class TemplateAnalyzer():
    def __init__(self, params:dict, container:DataContainer) -> None:
        print(f'TestAnalyzer: available devices={container.available_devices}')

    def run(self):
        print('run called')