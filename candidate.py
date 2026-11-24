import csv

with open('enjoysport.csv') as f:
  data = list(csv.reader(f))

n = len(data[0])-1
S = ['0']*n
G = ['?']*n

for row in data[1:]:
  x,y = row[:-1], row[-1]

  if y == 'yes':
    for j in range(n):
      if S[j] == '0':
        S[j] = x[j]
      elif S[j]!=x[j]:
        S[j] = '?'

  else:
    for j in range(n):
      if G[j]=='?' and S[j]!=x[j]:
        G[j] = S[j]

print(S)
print(G)