import pickle
if __name__ == '__main__':
    x={'xfg':1,'wjc':234}
    d=pickle.dumps(x)
    y=str(d)
    j=str.encode(y)
    l=pickle.loads(d)
    print(l)
    pass