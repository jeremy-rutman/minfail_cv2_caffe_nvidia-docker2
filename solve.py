__author__ = 'jeremy'
import caffe
import surgery, score

import numpy as np
import os
import sys

import setproctitle
import subprocess

from trendi.classifier_stuff.caffe_nns import jrinfer
from trendi.classifier_stuff.caffe_nns import progress_plot

setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = '../vgg16fc.caffemodel'  #cannot find this
weights = 'snapshot/train_iter_1799.caffemodel'
weights = 'snapshot/train_iter_359310.caffemodel'
weights = 'snapshot/train_iter_370000.caffemodel'
weights = 'VGG_ILSVRC_16_layers.caffemodel'
weights = 'snapshot/train_0816__iter_25000.caffemodel'  #in brainia container jr2

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
all_layers = [k for k in solver.net.params.keys()]
print('all layers:')
print all_layers
surgery.interp(solver.net, interp_layers)

# scoring
#val = np.loadtxt('../data/segvalid11.txt', dtype=str)
val = range(0,1500)

#jrinfer.seg_tests(solver, False, val, layer='score')
progress_plot.parse_solveoutput('net_output.txt')
cmd = 'scp  net_output.txt.jpg root@104.155.22.95:/var/www/results/progress_plots/brainik_pixlevel.jpg';
subprocess.call(cmd,shell=True)

for _ in range(1000):
    solver.step(5000)
#    score.seg_tests(solver, False, val, layer='score')
    jrinfer.seg_tests(solver, False, val, layer='score')
    progress_plot.parse_solveoutput('net_output.txt')
    subprocess.call(cmd,shell=True)
