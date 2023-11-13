import os
import sys

paths = ['../results/10ss/allReqs/',
         '../results/10ss/req0/',
         '../results/10ss/req1/',
         '../results/10ss/req2/',
         '../results/10ss/req3/',
         '../results/1ss/allReqs/',
         '../results/1ss/req0/',
         '../results/1ss/req1/',
         '../results/1ss/req2/',
         '../results/1ss/req3/']

os.chdir(sys.path[0])

for path in paths:
    os.system('python resultAnalyzer.py ' + path)
