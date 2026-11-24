import csv

with open('enjoysport.csv') as f:
  data = list(csv.reader(f))

n = len(data[0])-1
hypothesis = ['0']*n

for row in data[1:]:
  if row[-1].strip().lower() == "yes":
    for j in range(n):
      if hypothesis[j] == '0':
        hypothesis[j] = row[j]
      elif hypothesis[j]!=row[j]:
        hypothesis[j] = '?'

print(hypothesis)