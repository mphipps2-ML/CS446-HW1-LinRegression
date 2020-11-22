import torch

##############################
## Specify the matrix X
## Dimensions: X (3x3)
##############################
X = torch.Tensor([[0,0,1],[1,1,1],[4,2,1]])
y = torch.Tensor([[0],[1],[1]])
print(X)
print(y)

# Solution
##############################
## Use one of the ways to compute the result
#################e#############
res1 = torch.lstsq(y, X)
print("Solution 1:")
print(res1[0])

l = torch.matmul(torch.transpose(X, 0, 1),X)
r = torch.matmul(torch.transpose(X, 0, 1),y)
res2 = torch.solve(r,l)
print("Solution 2:")
print(res2[0])

l = torch.matmul(torch.transpose(X, 0, 1),X)
r = torch.matmul(torch.transpose(X, 0, 1),y)
res3 = torch.matmul(torch.inverse(l),r)
print("Solution 3:")
print(res3)
