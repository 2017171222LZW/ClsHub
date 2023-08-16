import pandas as pd

if __name__ == '__main__':
    # 数据框类型举例：
    text = []
    with open('output/rocks/adacanet_base_d/log.txt', 'r') as f:
        while True:
            r = f.readline()
            if r == '' or r == '\n':
                break
            text.append(eval(r))
            print(r)
    df = pd.DataFrame(text, columns=list(text[0].keys()))
    print(df.head(10))
    df.to_csv('./adacanet_base_d.csv')
