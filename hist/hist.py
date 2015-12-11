import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
import sys

from matplotlib.backends.backend_pdf import PdfPages

print sys.argv[1]
file=open(sys.argv[1])
lines=[line.split() for line in file]
x=[int(line[0]) for line in lines]
#y=[int(line[1]) for line in lines]
#H, edges = np.histogram(x)
x_hist, bins = np.histogram(x, 100)
plt.plot(bins[:-1], x_hist)
plt.xscale('log')
plt.yscale('log')
#plt.hist(x, 100)
fig = plt.gcf()
fig.savefig(sys.argv[2])
#fig = plt.figure()
#pp = PdfPages(sys.argv[2])
#pp.savefig(fig)
#H, xedges, yedges = np.histogram2d(y, x, (range(1, 194), range(1, 194)))
#for i in H:
  #for j in i:
    #print j,
  #print
#exit()
