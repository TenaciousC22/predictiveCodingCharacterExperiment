import csv

lines=[]
with open("predictiveCodingCharacterResults.csv") as csvFile:
	reader=csv.reader(csvFile, delimiter=",")

	for row in reader:
		lines.append(row)

newLines=[]
for line in lines:
	speaker=line[0]
	clip=line[1]
	offset=line[2]
	string=line[3]
	for x in range(4,len(line)):
		string="".join((string,(line[x])))

	newLines.append([speaker,clip,offset,string])

with open("predictiveCodingChar2String.csv","w") as file:
	writer=csv.writer(file,delimiter=',')
	writer.writerows(newLines)