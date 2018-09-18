#!/usr/bin/python3

data=[]

with open('data.csv')  as  f:
	x=f.read().split('\t')
	for row  in  x:
		data.append(row)


print(data)

