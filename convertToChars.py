import os
import numpy as np
from tqdm import tqdm

def splitString(base):
	split=base[7:-1]
	return split
	#print(split)

char2index={" ":1, "'":22, "1":30, "0":29, "3":37, "2":32, "5":34, "4":38, "7":36, "6":35, "9":31, "8":33,
	"A":5, "C":17, "B":20, "E":2, "D":12, "G":16, "F":19, "I":6, "H":9, "K":24, "J":25, "M":18,
	"L":11, "O":4, "N":7, "Q":27, "P":21, "S":8, "R":10, "U":13, "T":3, "W":15, "V":23, "Y":14,
	"X":26, "Z":28, "<EOS>":39}    #character to index mapping
index2char={1:" ", 22:"'", 30:"1", 29:"0", 37:"3", 32:"2", 34:"5", 38:"4", 36:"7", 35:"6", 31:"9", 33:"8",
	5:"A", 17:"C", 20:"B", 2:"E", 12:"D", 16:"G", 19:"F", 6:"I", 9:"H", 24:"K", 25:"J", 18:"M",
	11:"L", 4:"O", 7:"N", 27:"Q", 21:"P", 8:"S", 10:"R", 13:"U", 3:"T", 15:"W", 23:"V", 14:"Y",
	26:"X", 28:"Z", 39:"<EOS>"}    #index to character reverse mapping

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
		#print(f.read())