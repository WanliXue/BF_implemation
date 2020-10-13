import os
import pandas
import numpy as np


class Load_TestData:
    def __init__(self,name):

        self.name = name
        self.data=[]

    def load_data(self):
        database = '..' + os.sep + 'data' + os.sep + '%s.txt' % self.name
        series = pandas.read_csv(database, header=-1, delimiter='\t')
        t_data = np.array(series, dtype=pandas.Series)
        data=t_data
        for i in range(len(t_data)):
            data = t_data[i][0].split(' ')
            # data = map(float, data)
            data.append(filter(None, data))
            # print len(data_har)
            # print i

        data = np.array(data, dtype=np.float)
        return data