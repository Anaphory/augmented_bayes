import numpy
import operator
from functools import reduce

normalize = lambda x: x/x.sum(0)

class Node:
    nextindex = 0
    def __init__(self,
                 probabilities,
                 parents,
                 name=None,
                 labels=None):
        if name is None:
            self.name = Node.nextindex
            Node.nextindex += 1
        else:
            self.name=name
        self.parents = parents
        i = 0
        self.probabilities = numpy.asarray(probabilities)
        for parent in parents:
            assert(len(parent) == self.probabilities.shape[i])
            parent.lambdas[self] = numpy.ones(len(parent.labels))
            i = i+1
        self.c = i
        #I could probably just use -1, but I may want to use multiple
        #models in parallel later, and then it's nice to have the
        #least relevant dimensions as the last one.
        if labels is None:
            self.labels = range(self.probabilities.shape[self.c])
        else:
            assert(len(labels)==self.probabilities.shape[self.c])
            self.labels=labels
            
        self.lambdas = {}
        self.pis = [parent.belief() for parent in self.parents]

    def __len__(self):
        return self.probabilities.shape[self.c]


    def belief(self):
        pi = self.probabilities
        for parent_pi in self.pis:
            pi = numpy.tensordot(parent_pi, pi, 1)

        self._belief = normalize(
            reduce(operator.mul, self.lambdas.values(), numpy.ones(self.probabilities.shape[self.c]))
            *pi)
        return self._belief

    def up(self):
        lmbda = reduce(operator.mul, self.lambdas.values(), numpy.ones(self.probabilities.shape[self.c]))

        for parent in self.parents:
            pi = self.probabilities
            for i, other_parent in enumerate(self.parents):
                if parent != other_parent:
                    pi = numpy.tensordot(self.pis[i], pi, 1)
                else:
                    pi = numpy.rollaxis(pi, 0, -1)

            parent.lambdas[self] = normalize(numpy.tensordot(pi, lmbda, 1))

    def down(self):
        p = self.probabilities
        for pi in self.pis:
            p = numpy.tensordot(pi, p, 1)

        for child in self.lambdas.keys():
            if child != "evidence":
                me = child.parents.index(self)
            else:
                continue
            lmbda = 1
            for other_child, value in self.lambdas.items():
                if child != other_child:
                    lmbda = value * lmbda
            
            child.pis[me] = normalize(lmbda*p)
    
    def __repr__(self):
        return "<Node %s>" %(self.name)
        

class TreeAugmentedClassifier:
    def __init__(self, classes, a_priori_probabilities):
        self.classes = Node(a_priori_probabilities, [], "class", classes)
        self.nodes = []
        
    def add_node(self, p, name, labels=None, parent=None):
        #p: conditional_probabilities_by_class
        p = numpy.asarray(p)
        if parent:
            cl, pa, pr = p.shape
            assert(cl == len(self.classes.labels))
            assert(pa == len(parent.labels))
            new = Node(p, [self.classes, parent], name, labels)
        else:
            cl, pr = p.shape
            assert(cl == len(self.classes.labels))
            new = Node(p, [self.classes], name, labels)
        self.nodes.append(new)
        return new
    
    def inference(self):
        self.classes.down()
        for node in self.nodes:
            node.down()
        for node in reversed(self.nodes):
            node.up()
        self.classes.down()
        return self.classes.belief()

