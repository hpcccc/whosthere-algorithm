import pandas as pd
import warnings
import random
import operator
warnings.filterwarnings('ignore')

class SimpleTagBased:
    def __init__(self, ratio, data_path):
        self.ratio = ratio
        self.data_path = data_path
        self.records = {}
        self.train_data = {}
        self.test_data = {}
        self.user_tags = dict()
        self.user_items = dict()
        self.tag_items = dict()
        self.load_data()
        self.train_test_split()
        self.initStat()

    def load_data(self):
        print('数据正在加载中...')
        df = pd.read_csv(self.data_path,sep = '\t')
        #将df放入设定的字典格式中
        for i in range(len(df)):
        #for i in range(10):
            uid = df['userID'][i]
            iid = df['bookmarkID'][i]
            tag = df['tagID'][i]
            #setdefault将uid设置为字典，iid设置为[]
            self.records.setdefault(uid,{})
            self.records[uid].setdefault(iid,[])
            self.records[uid][iid].append(tag)
        #print(records)
        print('数据集大小为：%d.' %len(df))
        print('设置tag的人数:%d.' %len(self.records))
        print('数据加载完成\n')

    #将数据集拆分为训练集及测试集,ratio为测试集划分比例
    def train_test_split(self, seed = 100):
        random.seed(seed)
        for u in self.records.keys():
            for i in self.records[u].keys():
                #ratio为设置的比例
                if random.random()<self.ratio:
                    self.test_data.setdefault(u,{})
                    self.test_data[u].setdefault(i,[])
                    for t in self.records[u][i]:
                        self.test_data[u][i].append(t)
                else:
                    self.train_data.setdefault(u,{})
                    self.train_data[u].setdefault(i,[])
                    for t in self.records[u][i]:
                        self.train_data[u][i].append(t)
        print("训练集user数为：%d，测试机user数为：%d." % (len(self.train_data),len(self.test_data)))

    #设置矩阵mat[index,item]，来储存index与item 的关系, = {index:{item:n}},n为样本个数
    def addValueToMat(self,mat,index,item,value = 1):
        if index not in mat:
            mat.setdefault(index,{})
            mat[index].setdefault(item,value)
        else:
            if item not in mat[index]:
                mat[index].setdefault(item,value)
            else:
                mat[index][item] +=value

    #使用训练集,初始化user_tags,user_items,tag_items，/user_tags为{user1：{tags1:n}}
    #{user1：{tags2:n}}...{user2：{tags1:n}}，{user2：{tags2:n}}....n为样本个数等
    # user_items为{user1:{items1:n}}......原理同上
    # tag_items为{tag1:{items1:n}}......原理同上
                
    def initStat(self):
        records = self.train_data
        for u,items in records.items():
            for i,tags in records[u].items():
                for tag in tags:
                    #users和tag的关系矩阵
                    self.addValueToMat(self.user_tags,u,tag,1)
                    #users和item的关系
                    self.addValueToMat(self.user_items,u,i,1)
                    #tag和item的关系
                    self.addValueToMat(self.tag_items,tag,i,1)
        print('user_tags,user_items,tag_items初始化完成.')

    #对某一用户user进行topN推荐
    def recommend(self, user, N):
        recommend_item = dict()
        tagged_items = self.user_items[user]
        for tag,utn in self.user_tags[user].items():
            for item,tin in self.tag_items[tag].items():
                #如果某一user已经给某一item打过标签，则不推荐
                if item in tagged_items:
                    continue
                if item not in recommend_item:
                    recommend_item[item] = utn * tin
                else:
                    recommend_item[item] = recommend_item[item]+utn*tin
        #按value值，从大到小排序
        return sorted(recommend_item.items(), key=operator.itemgetter(1), reverse=True)[0:N]

    #使用测试集，计算准确率和召回率
    def precisionAndRecall(self, N):
        hit = 0
        h_recall = 0
        h_precision = 0
        for user,items in self.test_data.items():
            if user not in self.train_data:
                continue
            rank = self.recommend(user,N)
            for item,rui in rank:
                if item in items:
                    hit = hit+1
            h_recall = h_recall +len(items)
            h_precision = h_precision+N

        #返回准确率和召回率
        return (hit/(h_precision*1.0)), (hit/(h_recall*1.0))

    #使用test_data对推荐结果进行评估
    def testRecommend(self):
        print('推荐结果评估如下：')
        print("%3s %10s %10s" % ('N', "精确率", '召回率'))
        for n in [5,10,20,40,60,80,100]:
            precision,recall = self.precisionAndRecall(n)
            print("%3d %10.3f%% %10.3f%%" % (n, precision * 100, recall * 100))


if __name__ == '__main__':
    
    data_path = '/Users/v_hongpengcheng/Downloads/hetrec2011-delicious-2k/user_taggedbookmarks-timestamps.dat'
    simpletagbased = SimpleTagBased(ratio = 0.2, data_path = data_path)
    print(simpletagbased.recommend(8, 10))