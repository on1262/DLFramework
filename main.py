from analyzers.analyzer import Analyzer
import yaml
from tools import logger

if __name__ == '__main__':
    # load launch configs
    with open('launch_conf.yml', 'r', encoding='utf-8') as fp:
        analyzer_list = yaml.load(fp, Loader=yaml.SafeLoader)['analyzer_list']
        assert(isinstance(analyzer_list, list))
    
    if len(analyzer_list) > 0:
        analyzer = Analyzer(analyzer_list)
    logger.info('Main.py: Done')