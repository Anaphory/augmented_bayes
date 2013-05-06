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

tan = TreeAugmentedClassifier(
    ["A","B"],
    [0.9, 0.1])

position = tan.add_node([[0.5,0.4,0.1],[0.1,0.3,0.6]], "position", ["front","mid","end"])
tan.add_node([[[.7,.2,.1],[.2,.6,.2],[.1,.2,.7]],[[.8,.1,.1],[.6,.3,.1],[.4,.4,.2]]], "priority", ["high","mid","low"], position)
tan.add_node([[0.5,0.5],[0.1,0.9]], "weight", ["light","heavy"])

tan.nodes[1].lambdas["evidence"]=[1,0,0]
print(tan.inference())
