import mlp, numpy as np

def main():
    dataset = [((0,0),0),((0,1),1),((1,0),1),((1,1),0)]
    
    m = mlp.MLP([2, 2, 1])
    m.train(dataset)

if __name__ == '__main__':
    main()
