import os
import numpy as np
from tqdm import tqdm

def splitString(base):
	split=base[7:-1]
	return split
	#print(split)

char2index={" ":0, "'":21, "1":29, "0":28, "3":36, "2":31, "5":33, "4":37, "7":35, "6":34, "9":30, "8":32,
	"A":4, "C":16, "B":19, "E":1, "D":11, "G":15, "F":18, "I":5, "H":8, "K":23, "J":24, "M":17,
	"L":10, "O":3, "N":6, "Q":26, "P":20, "S":7, "R":9, "U":12, "T":2, "W":14, "V":22, "Y":13,
	"X":25, "Z":27}    #character to index mapping
index2char={0:" ", 21:"'", 29:"1", 28:"0", 36:"3", 31:"2", 33:"5", 37:"4", 35:"7", 34:"6", 30:"9", 32:"8",
	4:"A", 16:"C", 19:"B", 1:"E", 11:"D", 15:"G", 18:"F", 5:"I", 8:"H", 23:"K", 24:"J", 17:"M",
	10:"L", 3:"O", 6:"N", 26:"Q", 20:"P", 7:"S", 9:"R", 12:"U", 2:"T", 14:"W", 22:"V", 13:"Y",
	25:"X", 27:"Z"}    #index to character reverse mapping

base_dir="/home/analysis/Documents/studentHDD/datasets/LRS2/mvlrs_v1/main"

dirs=os.listdir(base_dir)

x=0
while x < len(dirs):
	if ".wav" in dirs[x]:
		dirs.pop(x)
	else:
		x+=1

for entry in tqdm(dirs):
	path=base_dir+"/"+entry
	dirWalk=os.walk(path)
	for walkEntry in dirWalk:
		files=walkEntry[2]
	
	x=0
	while x < len(files):
		if ".txt" not in files[x]:
			files.pop(x)
		elif "words" in files[x]:
			files.pop(x)
		else:
			x+=1

	for file in files:
		openable=path+"/"+file
		f=open(openable,"r")
		text=splitString(f.readline())
		f.close()
		outputName=path+"/"+file[0:5]+"-words.txt"
		f=open(outputName,"w")
		f.write(text)
		f.close()
		charIndex=[]
		for x in range(len(text)):
			charIndex.append(char2index[text[x]])
		arr=np.array(charIndex)
		outputName=path+"/"+file[0:5]+"-chars.npy"
		np.save(outputName,arr)
		#print(f.read())