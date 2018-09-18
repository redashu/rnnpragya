#!/usr/bin/python3

import  csv
import  matplotlib.pyplot as plt
data=[]

with open('data.csv')  as  f:
	readf=csv.reader(f,delimiter='\t',quoting=csv.QUOTE_NONE)
	for row  in  readf:
		data.append(row)


print(data)

with  open('new.txt','w') as  f:
	for  i  in  data:
		value=i[0]+','+i[1].replace(',','.')+'\n'
		if  'X' in value or 'Y' in value:
			continue 
		else :

			f.write(value)




