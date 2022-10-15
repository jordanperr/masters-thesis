import sys
import pandas as pd

path=sys.argv[1]

if __name__=="__main__":
    experiments = pd.read_csv(path+"/index.csv").to_dict("records")