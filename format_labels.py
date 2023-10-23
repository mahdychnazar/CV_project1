import pandas as pd
import os


columns = ['file','xmin','ymin','xmax','ymax']
def format_Labels():
    boxes = pd.read_csv('F:/CV/_annotations.txt', sep="\s+| |, ", names=columns, header=None,  engine="python")
    boxes.to_csv('annotation.csv')
    return



if __name__ == "__main__":
    format_Labels()
