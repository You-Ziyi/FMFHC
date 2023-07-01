
import pandas as pd

pulsar = pd.read_csv(r'*/res1.csv',encoding='gbk')
pulsar=pulsar.iloc[:,:]



nonpulsar = pd.read_csv(r'*/res2.csv',index_col=False,encoding='gbk')
nonpulsar = nonpulsar.iloc[:, :]

chunk = 20
len_nonpulsar = len(nonpulsar)
shumu=round(len_nonpulsar/chunk)

length1 = shumu
length2 = 2*shumu

yesnpulsar = pd.read_csv(r'*/res5.csv', encoding='gbk')
yesnpulsar = yesnpulsar.iloc[:,:]


i = 0
num = 0
while i < len_nonpulsar:

    if i+length1 == len_nonpulsar:
        data = pd.concat([pulsar, nonpulsar.iloc[i:i + length1, :], nonpulsar.iloc[0:length1, :], yesnpulsar.iloc[num:num+1, :]], axis=0)
    else:
        data = pd.concat([pulsar, nonpulsar.iloc[i:i + length2, :], yesnpulsar.iloc[num:num+1, :]], axis=0)

    outputPath = r'*/data/'+str(num)+'.csv'
    data.to_csv(outputPath, index=None)
    num+=1
    i+=(length2-length1)



