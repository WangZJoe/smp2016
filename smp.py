# -*- coding : UTF-8 -*-
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

provsString='''
    东北,辽宁,吉林,黑龙江
    华北,河北,山西,内蒙古,北京,天津
    华东,山东,江苏,安徽,浙江,台湾,福建,江西,上海
    华中,河南,湖北,湖南
    华南,广东,广西,海南,香港,澳门
    西南,云南,重庆,贵州,四川,西藏
    西北,新疆,陕西,宁夏,青海,甘肃
    境外,其他,海外
    None,None
    '''

clsaa Dataset():
    def __init__(self):
        self.provs={}
        self.offset = 0

        for line in provsString.split('\n'):
            items=line.split(',')
            for item in items[1:]:
                self.provs[item]=items[0].strip()

        #读取训练集
        train_file = map(lambda x:x.split(',',maxsplit=5),
                        open('./dataset/train/train_status.txt',encoding='utf8'))

        valid_labels = set(map(lambda x:x.strip(), open('./dataset/valid/valid_nolabel.txt')))
        valid_file = filter(lambda x:x[0] in valid_labels, 
                            map(lambda x:x.split(',',maxsplit=5), 
                            open('./dataset/valid/valid_status.txt',encoding='utf8')))

        df = pd.DataFrame(data=list(train_file)+list(valid_file),
                            columns='id,review,forward,source,time,content'.split(','),)

        # 读取训练集标注
        labels = pd.read_csv('./dataset/train/train_labels.txt',sep='\|\|',encoding='utf8',engine='python',
                              names='id,gender,age,location'.split(','))
        labels.age = labels.age.apply(self.__map_age__)
        labels.location = labels.location.apply(__map_location__)
        

        # 按id进行合并微博内容
        X = pd.DataFrame(df.groupby(by='id',sort=False).content.sum()).reset_index()
        X.id = X.id.astype(int)
        data = pd.merge(X,labels,on='id',how='left')

        # 从文本中提取TFIDF特征
        tfidf=TfidfVectorizer(max_features=512)
        self.f_tfidf=tfidf.fit_transform(data.content)

    def get_next_batch(batch_size):
        data = self.f_tfidf[self.offset:self.offset+batch_size]
        labels = np.concatenate((self.labels.age[self.offset:self.offset+batch_size],
                                self.labels.gender[self.offset:self.offset+batch_size],
                                self.labels.location[self.offset:self.offset+batch_size]))
        pass


 
    def __map_age__(x):
        x=int(x)
        if x>=1990:
            return '1990+'
        elif x<1980:
            return '1979-'
        else:
            return '1980-1989'
        
    def __map_location__(x):
        x=x.split(' ')[0]
        return self.provs[x]





# print(type(f_tfidf))
# print(f_tfidf.shape)
# print(f_tfidf[0].size)

# print(len(tfidf.get_feature_names()))
# print(len(tfidf.vocabulary_))
