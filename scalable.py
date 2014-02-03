from openmdao.main.api import Component, Assembly
from openmdao.main.datatypes.api import Float, Array
from itertools import product, ifilter, compress
from scipy.misc import comb
import numpy as np

import time


total = 0

class Polynomial(Component):

    F = Float(0., iotype="out")

    def __init__(self, m=3, p=3):
        self.m = m
        self.p = p
        super(Polynomial, self).__init__()
        self.add("x", Array(np.zeros(m), iotype="in"))
        #self.add("c", Array(np.ones(self.n_terms), iotype="in"))
        print "making indices..."
        tuples = product(xrange(p + 1), repeat=m)

        def filter_func(j): 
            global total 
            total += 1
            s = sum(j)

            return s > 0 and s <= p
        start = time.time()
        #self.dvals = list(ifilter(lambda j : sum(j) > 0 and sum(j) <= p, tuples))
        self.dvals = list(ifilter(filter_func, tuples))
        print "done making incicies: ", time.time() - start, total

    def execute(self):
        terms = {}
        #print "computing vals.."
        for i in self.dvals:
            other = [0]*self.m
            list_i = list(i)
            valmax = np.argmax(i)
            list_i[valmax] += -1
            other[valmax] = 1
            list_i, other = tuple(list_i), tuple(other)
            if list_i in terms and other in terms:
                term = terms[list_i] * terms[other]
            else:
                term = np.prod([self.x[k]**v for k,v in enumerate(i) if v])
            terms[i] = term

if __name__ == "__main__":
    p = Polynomial(10,4)
    p.x = np.arange(1, 11, dtype=np.float32)
    print len(p.dvals)

    start = time.time()
    for i in xrange(1000): 
        p.run()
    print (time.time() - start)/1000.