# -*- coding : UTF-8 -*-
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
provs={}

for line in provsString.split('\n'):
    items=line.split(',')
    for item in items[1:]:
        provs[item]=items[0].strip()
    
def map_age(x):
    x=int(x)
    if x>=1990:
        return '1990+'
    elif x<1980:
        return '1979-'
    else:
        return '1980-1989'
    
def map_location(x):
    x=x.split(' ')[0]
    return provs[x]

import pandas as pd
#读取训练集
train_file = map(lambda x:x.split(',',maxsplit=5),
               open('train/train_status.txt',encoding='utf8'))

valid_labels = set(map(lambda x:x.strip(), open('valid/valid_nolabel.txt')))
valid_file = filter(lambda x:x[0] in valid_labels, 
                map(lambda x:x.split(',',maxsplit=5), 
                    open('valid/valid_status.txt',encoding='utf8')))

df = pd.DataFrame(data=list(train_file)+list(valid_file),
                columns='id,review,forward,source,time,content'.split(','),)

print(df.head(5))
#读取训练集标注
#labels = pd.read_csv('train/train_labels.txt',sep='\|\|',encoding='utf8',engine='python',
#                   names='id,gender,age,location'.split(','))

#labels.age = labels.age.apply(map_age)
#labels.location = labels.location.apply(map_location)

#按id进行合并微博内容
#X = pd.DataFrame(df.groupby(by='id',sort=False).content.sum()).reset_index()
#X.id = X.id.astype(int)

#data = pd.merge(X,labels,on='id',how='left')

#if __name__ == '__main__' :
#    main()
