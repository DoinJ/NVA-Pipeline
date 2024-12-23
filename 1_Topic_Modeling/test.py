import pandas as pd

if __name__ == '__main__':
    # 读取数据
    df = pd.read_csv('data/topic_info.csv')
    # 查看数据
    for idx, line in df.iterrows():
        Name, KeyBERT, Llama3 = line['Name'], line['KeyBERT'], eval(line['Llama3'])[0]
        print(Name, KeyBERT, Llama3)
