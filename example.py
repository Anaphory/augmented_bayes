from inference import Node

U = Node([0.9, 0.1], [])
V = Node([[1, 0], [0.5, 0.5]], [U])
X = Node([0.4, 0.2, 0.4], [])
Y = Node([[[1,0,0],[0.5,0.5,0],[0,1,0]],[[0,0.5,0.5],[0,0,0],[0,0,1]]], [U,X])

V.lambdas["evidence"] = numpy.array([0,1])
print(V.belief())
V.up()
print(U.belief())
U.down()
print(Y.belief())
Y.up()
print(X.belief())
