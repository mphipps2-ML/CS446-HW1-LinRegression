import torch

##############################
## Specify the matrix X
## Dimensions: X (3x3)
##############################
X = torch.Tensor([[0,1,4],[0,1,2],[1,1,1]])
y = torch.Tensor([[0],[1],[1]])
print(X)
print(y)

# Solution
##############################
## Use one of the ways to compute the result
##############################
res1 = torch.lstsq(y,X)
print(res1[0])
