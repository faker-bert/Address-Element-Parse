import pandas
import predict


data = pandas.read_excel('企业注册地址与装机地址匹配验证.xlsx').iloc[:, -1].tolist()
# print(data)
for i in data:
    if type(i) is float:
        continue
    if '福建省' not in i and '福建' in i:
        i = i.replace('福建', '福建省')
    i = i.replace("|", "").replace("/", "").replace("-", "")
    predict.predict(i)

