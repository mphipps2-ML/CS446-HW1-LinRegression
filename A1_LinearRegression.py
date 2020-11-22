import torch

x = torch.Tensor([[1],[2]])
y = torch.Tensor([[1],[1]])
print(x.size())

X = torch.cat((x, torch.ones_like(x)),1)
print(X)
print(X[1,0])
print(torch.matmul(X, y))

# Solution 1
##############################
## Fill in the arguments
##############################
#gels deprecated for lstsq
#res1 = torch.gels(y, X)
#solves min|Xw - Y|
# easiest way dummy-proof way of solving least squares linear regression problem
res1 = torch.lstsq(y, X)
print("Solution 1:")
#torch.lstsq(y,X) returns (Tensor,Tensor); first argument is solution; second is details of the QR factorization 
print(res1[0])
#print(res1)

# Solution 2
##############################
## How to compute l and r?
## Dimensions: l (2x2); r (2x1)
##############################
# torch.solve returns the solution to system of linear equations represented by Xw = y, ie w* = (X^-1 y)
# allows us to solve starting form w* = (X^T X)^-1 X^T Y
# don't need to directly input torch.inverse(l) using this method

print("X ", X)
print("y ", y)
print("torch.matmul(torch.transpose(X, 0, 1),X) " ,torch.matmul(torch.transpose(X, 0, 1),X))
print("torch.matmul(torch.transpose(X, 0, 1),y) ", torch.matmul(torch.transpose(X, 0, 1),y))

# l = (X^TX) = torch.Tensor([[5,3],[3,2]])
l = torch.matmul(torch.transpose(X, 0, 1),X)
# r = (X^TY) = torch.Tensor([[3],[2]])
r = torch.matmul(torch.transpose(X, 0, 1),y)

# gesv deprecated for solve
#res2 = torch.gesv(r,l)
res2 = torch.solve(r,l)

print("Solution 2:")
#torch.solve(r,l) returns (tensor,tensor) with first argument being solution; second being LU factorization
#print(res2[0])
print(res2)

# Solution 3
##############################
## What is l and r?
## Dimensions: l (2x2); r (2x1)
##############################
# using torch.matmul is direct matrix multiplication which requires extra step of specifying torch.inverse(l)
res3 = torch.matmul(torch.inverse(l),r)
print("Solution 3:")
print(res3)

