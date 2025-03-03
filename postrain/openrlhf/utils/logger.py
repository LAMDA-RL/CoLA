import numpy as np

class Logger():

    def __init__(self) -> None:
        self.log_dict = {}
        self.global_step = 0

    def add(self, data_dict, step=None):
        if 'steps' in data_dict.keys():
            steps = data_dict['steps']
        else:
            steps = self.global_step

        for key in data_dict.keys():
            if key not in self.log_dict.keys():
                self.log_dict[key] = []
            
            self.log_dict[key].append((steps, data_dict[key]))
        
        self.global_step += 1

    def save(self, path):
        np.save(path, self.log_dict)