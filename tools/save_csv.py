import pandas as pd
from pathlib import Path

# the dir-path of log.txt
result_dir = '../output/nimrf_10c/ghostnet'


def save_csv(dir):
    if Path(dir + '/result.csv').exists():
        print("file is existed.")
        return
    text = []
    with open(dir + '/log.txt', 'r') as f:
        while True:
            r = f.readline()
            if r == '' or r == '\n':
                break
            text.append(eval(r))
            print(r)
    df = pd.DataFrame(text, columns=list(text[0].keys()))
    print(df.head(10))
    df.to_csv(dir + '/result.csv')
    print('ok')


if __name__ == '__main__':
    save_csv(result_dir)
