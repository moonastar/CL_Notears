from other import sEM
import pandas
from pandas.core.frame import DataFrame
import openpyxl
import threading

# 导入数据
class MyThread(threading.Thread):
    def __init__(self, func, args, name=''):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func
        self.args = args
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


tables = []
threas1 = []
network = ['asia','sachs','child','alarm','hailfinder','hepar2','andes']
paths = ["C:\\Users\\华为\\Desktop\\data\\cl.xlsx",
         "C:\\Users\\华为\\Desktop\\data\\notears.xlsx"]
for e in range(0,2):
    for sample in [1000,2000,5000]:
        for i in range(0,5):
            s = int(sample)
            print("the network:", s)
            th = MyThread(sEM, (sample,'gauss',network[e]), sEM.__name__)
            threas1.append(th)
for i in range(0, len(threas1)):
    threas1[i].start()
    threas1[i].join()
    tab = threas1[i].get_result()
    tables.append(tab)
    ta = DataFrame(tables)
    ta.rename(columns={0:'network',1: 'f1-score', 2: 'Precision', 3: 'FDR', 4: 'TPR', 5: 'FPR',
                       6: 'SHD', 7: 'NNZ', 8: 'TP'}, inplace=True)
    book = openpyxl.load_workbook(paths[1])
    sheet = book.active
    w = pandas.ExcelFile(paths[1], engine='openpyxl')
    ta.to_excel(w)
