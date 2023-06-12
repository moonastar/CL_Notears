from other_dataset import CL_Notears,anCL_Notears
from curriculum import curr
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
        "C:\\Users\\华为\\Desktop\\data\\child.xlsx",
        "C:\\Users\\华为\\Desktop\\data\\alarm.xlsx",
"C:\\Users\\华为\\Desktop\\data\\hailfinder.xlsx",
"C:\\Users\\华为\\Desktop\\data\\hepar2.xlsx",
         "C:\\Users\\华为\\Desktop\\data\\notears.xlsx"]
for e in range(3,4):
    #curri = curr(network[e])
    for sample in [1750,2250]:
        for i in range(0,1):
            s = int(sample)
            print("the network:", s)
            th = MyThread(CL_Notears, (sample,'gauss',network[e]), CL_Notears.__name__)
            threas1.append(th)
for i in range(0, len(threas1)):
    threas1[i].start()
    threas1[i].join()
    tab = threas1[i].get_result()
    tables.append(tab)
    ta = DataFrame(tables)
    ta.rename(columns={0:'network',1: 'f1-score', 2: 'Precision', 3: 'FDR', 4: 'TPR', 5: 'FPR',
                       6: 'SHD', 7: 'NNZ', 8: 'TP'}, inplace=True)
    book = openpyxl.load_workbook(paths[0])
    sheet = book.active
    w = pandas.ExcelFile(paths[0], engine='openpyxl')
    ta.to_excel(w)

