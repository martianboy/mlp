#!/usr/bin/python
__author__ = 'Abbas Mashayekh, abbas.m@abbas-m.com'

import csv

def encodeCategory(cat):
    

def main():
    dataset = []
    with open('../EX1/data/hayes-roth.data', 'rb') as datafile:
        dsreader = csv.reader(datafile)
        for record in dsreader:
            dataset.append(((int(r[2]),int(r[3]),int(r[4])),int(r[5])))
            
    mlp = MLP([3, 3, 3])
    mlp.train(dataset)


if __name__ == "__main__":
	main()
