from analyzer import Analyzer

analyzer_params = [

]

if __name__ == '__main__':
    dataset = None # TODO
    if len(analyzer_params) > 0:
        analyzer = Analyzer(analyzer_params, dataset)
    print('Done')