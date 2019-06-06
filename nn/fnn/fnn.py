
# coding: utf-8

# In[1]:


#%source bin/activate
import tensorflow as tf
import os
import random
import time
import numpy as np
import sys
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[2]:


class runrecord:
    def __init__(self,
                trnlist,
                testlist,
                bThetas,
                sizes,
                batchsize,
                eta,
                keepprob,
                nEpoch,
                Nx):
        self.params = {
            "trnlist":trnlist,
            "testlist":testlist,
            "bThetas":bThetas,
            "sizes":sizes,
            "batchsize":batchsize,
            "eta":eta,
            "keepprob":keepprob,
            "nEpoch":nEpoch,
            "Nx":Nx
        }
        
        self.testacc = np.zeros(shape=(nEpoch,))        
        self.trnacc = np.zeros(shape=(nEpoch,))        
        self.testlosses = np.zeros(shape=(nEpoch,))        
        self.trnlosses = np.zeros(shape=(nEpoch,))


# In[11]:


rundir = "/home/walterms/project/walterms/mcmd/nn/data/"
trndir = rundir+"train/"
testdir = rundir+"test/"
bThetas = True
NLBL = 6

trnlist = ["xjam", "tjam", "edge15.00","ujam"]
# trnlist = ["edge15.00", "edge30.00"]
testlist = ["xjam", "tjam", "edge15.00","ujam"]
# testlist = ["edge15.00", "edge30.00"]

if trnlist == [""]:
    trnlist = [trndir+x for x in os.listdir(trndir)]
else:
    trnlist = [trndir+x for x in trnlist]
if testlist == [""]:
    testlist = [testdir+x for x in os.listdir(testdir)]
else:
    testlist = [testdir+x for x in testlist]


# In[26]:


def get_trndata(Nx=1):
    cellwidth = 1./Nx

    # Compile training set
    trn_data_ = []
    trn_inputs = []
    trn_labels = []

    nTrnPerFile = 3000
    shufIdx = [p for p in range(400)]
    # random.shuffle(shufIdx)

    for f in trnlist:
        print "Processing " + f + " as training data"
        thetas = []
        xs = []
        ys = []
        fin = open(f, 'r')
        nTrn = 0
        for line in fin.readlines():
            if line == "\n": continue
            if line.startswith("label"):
                # Done a block
                lbl = int(float(line.split()[1]))
                trn_labels.append(lbl)

                # Insert data
                random.shuffle(shufIdx)
                formatted_data = []
                cells = [[] for _ in range(Nx*Nx)]
                for s in shufIdx:
                    ic = int((xs[s]+0.5)//cellwidth + ((ys[s]+0.5)//cellwidth)*Nx)
                    cells[ic].append([xs[s],ys[s],thetas[s]])
                for cell in cells:
                    for rod in cell:
                        if bThetas:
                            formatted_data += [rod[2]]
                        else:
                            formatted_data += rod

                trn_inputs.append(formatted_data)

                thetas = []
                xs = []
                ys = []
                nTrn+=1
                if nTrn == nTrnPerFile: 
                    break
                continue
            spt = [float(x) for x in line.split()]
            xs.append(spt[0])
            ys.append(spt[1])
            thetas.append(spt[2])

        fin.close()

    for i in range(len(trn_inputs)):
        trn_data_.append([trn_inputs[i], trn_labels[i]])

    random.shuffle(trn_data_)
    print "Done compiling training set"

    return trn_data_


# In[21]:


trn_data = get_trndata(8)


# In[8]:


sys.getsizeof(trn_data)


# In[32]:


xs = [r for r in trn_data[0][0][1::3]]
plt.plot(xs)


# In[27]:


def get_testdata(Nx=1):
    cellwidth = 1./Nx
    test_data_ = []
    test_inputs = []
    test_labels = []

    nblPerFile = 300
    shufIdx = [ss for ss in range(400)]
    for f in testlist:
        print "Adding " + f + " to test set"
        thetas = []
        xs = []
        ys = []
        fin = open(f, 'r')
        nbl = 0
        for line in fin.readlines():
            if line == "\n": continue
            if line.startswith("label"):
                # Done a block
                lbl = int(float(line.split()[1]))
                test_labels.append(lbl)

                # Insert data
                random.shuffle(shufIdx)
                formatted_data = []
                cells = [[] for _ in range(Nx*Nx)]
                for s in shufIdx:
                    ic = int((xs[s]+0.5)//cellwidth + ((ys[s]+0.5)//cellwidth)*Nx)
                    cells[ic].append([xs[s],ys[s],thetas[s]])
#                     for s in shufIdx:
#                         formatted_data += [xs[s],ys[s],thetas[s]]
                for cell in cells:
                    for rod in cell:
                        if bThetas:
                            formatted_data += [rod[2]]
                        else:
                            formatted_data += rod
                test_inputs.append(formatted_data)
                thetas = []
                xs = []
                ys = []
                nbl+=1
                if nbl == nblPerFile: 
                    break
                continue
            spt = [float(x) for x in line.split()]
            xs.append(spt[0])
            ys.append(spt[1])
            thetas.append(spt[2])

        fin.close()
    for i in range(len(test_inputs)):
        test_data_.append([test_inputs[i], test_labels[i]])

    print "Done"
    return test_data_


# In[23]:


test_data = get_testdata()


# In[15]:


print len(trn_data), len(test_data)


# In[24]:


# Params
# labels -- [iso, D, T, X, U, L]
batchsize = 200
eta = .001

nInput = 1200
if bThetas: nInput = 400
sizes = [nInput,128,32,NLBL]

x_tf = tf.placeholder(tf.float32, shape=[None, nInput], name="x_tf")
y_ = tf.placeholder(tf.int64, shape=[None], name="y_")

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

keep_prob = tf.placeholder(tf.float32)

weights = []
biases = []
activs = []
activs.append(x_tf)

for s in range(len(sizes)-2):
    weights.append(weight_variable([sizes[s],sizes[s+1]]))
    biases.append(bias_variable([sizes[s+1]]))
    activs.append(tf.nn.dropout(tf.nn.elu(tf.matmul(activs[-1],weights[-1]) + biases[-1]), keep_prob))

# Last layer
weights.append(weight_variable([sizes[-2],sizes[-1]]))
biases.append(bias_variable([sizes[-1]]))
zout = tf.matmul(activs[-1],weights[-1]) + biases[-1]
# w4 = weight_variable([8,NLBL])
# b4 = bias_variable([NLBL])
# zout = tf.matmul(a3_drop,w4) + b4
y_out = tf.nn.softmax(zout)

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_,logits=zout))

train_step = tf.train.AdamOptimizer(learning_rate=eta).minimize(cost)
correct_prediction = tf.equal(tf.argmax(y_out, 1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print "Done"


# In[96]:


B = tf.placeholder(tf.int64, shape=[None,4])
C = tf.placeholder(tf.int64, shape=[None])
A = tf.equal(tf.argmax(B,1),C)
with tf.Session() as sess:
    y__ = np.asarray([0,1,1,0])
    yyy = np.asarray([[1,0,0,0],[0,1,0,0],[0,1,0,0],[1,0,0,0]])
    acc = sess.run([A],feed_dict={C:y__,B:yyy})
    print acc


# In[7]:


records = []


# In[116]:


# Remove all but last run from records
records = [records[-1]]


# In[12]:


# Remove most recent run from records
tmp = list(records)
records = []
for r in tmp[0:-1]:
    records.append(r)


# In[16]:


len(trn_data[0][0])


# In[31]:


nEpoch = 20
epocheval = 1
py_keepprob = .9
Nx = [1,2,4,6,8]

sess = tf.Session()
sess.run(tf.global_variables_initializer())
with sess.as_default():
    assert tf.get_default_session() is sess
# with tf.Session() as sess:
    # Gather performance on different grid divisions    
    
    for nx in Nx:
        sess.run(tf.global_variables_initializer())
        print "Training on Nx = "+str(nx)
        thisrecord = runrecord(trnlist,testlist,bThetas,sizes,batchsize,eta,py_keepprob,nEpoch,Nx=nx)
        records.append(thisrecord)
        
        # Gather data
        trn_data = get_trndata(nx)
        test_data = get_testdata(nx)
        
        xtest = np.asarray([test_data[i][0] for i in xrange(len(test_data))])
        ytest = np.asarray([test_data[i][1] for i in xrange(len(test_data))])

        for e in range(nEpoch):
            # Shuffle data
            random.shuffle(trn_data)

            for b in range(len(trn_data)/batchsize):
                ib = b*batchsize
                # Pepare a batch
                xin = np.asarray([trn_data[ib+iib][0] for iib in xrange(batchsize)])
                yin = np.asarray([trn_data[ib+iib][1] for iib in xrange(batchsize)])
                train_step.run(feed_dict={x_tf:xin, y_:yin, keep_prob:py_keepprob})

            if e % epocheval == 0:
                # Cost from trn data
                trnacc, trncost = sess.run([accuracy,cost],feed_dict={x_tf:xin, y_:yin, keep_prob:1.0})
                # Eval on test set
                random.shuffle(test_data)
                ntest = int(len(test_data)/3)
                xtest = np.asarray([test_data[i][0] for i in xrange(ntest)])
                ytest = np.asarray([test_data[i][1] for i in xrange(ntest)])
                test_accuracy, youts, testcost = sess.run([accuracy,y_out,cost],feed_dict={
                    x_tf: xtest, y_: ytest, keep_prob: 1.0})
                print('epoch %d, trnacc %g, test accuracy %g, trncost %g, testcost %g' 
                      % (e,trnacc,test_accuracy,trncost,testcost))
                thisrecord.testlosses[e] = testcost
                thisrecord.trnlosses[e] = trncost
                thisrecord.testacc[e] = test_accuracy
                thisrecord.trnacc[e] = trnacc
            
            
# sess.close()
print "Done"


# In[46]:


records[-1].params["Nx"]


# In[57]:


lennx = len(Nx)
colors = [(0.,(float(x)/lennx)*1.,0.4+(float(x)/lennx)*0.6) for x in range(lennx)]
f, ax = plt.subplots(1,2)
coldict = {}

for i in range(lennx):
    coldict.update({Nx[i]:colors[i]})
i = 0
for r in records[-lennx:]:
    nx = int(r.params["Nx"])
    x = np.argwhere(r.testlosses>0)
#     ax0, = ax[0].plot(x,r.trnlosses[x],'b-',label='train loss')
    ax0, = ax[0].plot(x,r.testlosses[x],color=coldict[nx],label=str(nx))
#     axtwin = ax[0].twinx()
#     ax0t, = axtwin.plot(x,r.testlosses[x],'g:',label='test loss')
#     l1 = plt.legend([ax0],["train loss", "test loss"])
    ax[1].plot(x,r.testacc[x],color=coldict[nx],label=str(nx))
    ax[1].legend()
    ax[0].set_yscale("log", nonposy='mask')
#     axtwin.set_yscale("log", nonposy='mask')

    i+=1
    
plt.gcf().set_size_inches(14,5)


# In[40]:


# Loss plot
# Mask on zeros
f, ax = plt.subplots(1,2)
for r in records[-1:]:
    x = np.argwhere(r.testlosses>0)
    ax0, = ax[0].plot(x,r.trnlosses[x],'b-',label='train loss')
    ax0, = ax[0].plot(x,r.testlosses[x],'g:',label='test loss')
#     axtwin = ax[0].twinx()
#     ax0t, = axtwin.plot(x,r.testlosses[x],'g:',label='test loss')
    l1 = plt.legend([ax0],["train loss", "test loss"])
    ax[1].plot(x,r.trnacc[x],'b-',label="train")
    ax[1].plot(x,r.testacc[x],'g:',label="test")
    ax[1].legend()
    ax[0].set_yscale("log", nonposy='mask')
#     axtwin.set_yscale("log", nonposy='mask')
    
plt.gcf().set_size_inches(14,5)


# In[21]:


edges = []
edgefile = open("../data/edgevar/edgelist_med","r")
for e in edgefile.readlines():
    edges.append(e.strip())
    
unlbl_append = "_rotnorm"
unlbldir = "/home/walterms/project/walterms/mcmd/nn/data/edgevar/unlabeled/"
unlbllist = {}
for e in edges:
    unlbllist.update({e:unlbldir+"edge"+str(e)+unlbl_append})


# In[30]:


with sess.as_default():   
    # Feed forward            
    # Gen predictions on unlabeled set
    maxSample = 500
    edgeOuts = {}
    for e in unlbllist:
        inputs = []
        print "Feeding " + e + " unlabeled set"
        thetas = []
        xs = []
        ys = []
        fin = open(unlbllist[e], 'r')
        nbl = 0
        for line in fin.readlines():
            if nbl==maxSample:
                break
            if line == "\n":
#                 # Done a block
                nbl+=1
                if bThetas:
                    random.shuffle(thetas)
                    inputs.append(thetas)
                else:
                    # Insert data as triplets
                    formatted_data = []
                    shufIdx = range(len(thetas))
                    random.shuffle(shufIdx)
                    for s in shufIdx:
                        formatted_data += [xs[s],ys[s],thetas[s]]
                    inputs.append(formatted_data)
                thetas = []
                xs = []
                ys = []
                continue
            spt = [float(s) for s in line.split()]
            xs.append(spt[0])
            ys.append(spt[1])
            thetas.append(spt[2])
        fin.close()
        # Feed forward
        xin = np.asarray(inputs)
        outs = y_out.eval(feed_dict={x_tf:xin, keep_prob:1.0})
        A = np.asarray(outs).transpose()
        edgeOuts.update({e:[np.average(A[0]), np.std(A[0])/np.sqrt(len(A[0])),
                            np.average(A[1]), np.std(A[1])/np.sqrt(len(A[1]))]})

print "Done"


# In[15]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[31]:


edgeOuts


# In[40]:


r0 = L*L*N/(23.19*23.19)
r1 = L*L*N/(23.63*23.63)
print r1, r0, (r1+r0)/2.


# In[44]:


xedge[0]


# In[32]:


N = 400
L = 3.0

xedge, xrho = [], []
avg0,avg1,std0,std1 = [],[],[],[]
x0,x1 = 0.,0. #locations of transition min/max
for e, out in sorted(edgeOuts.iteritems()):
    xedge.append(float(e))
    xrho.append(L*L*N/(float(e)*float(e)))
    avg0.append(out[0])
    std0.append(out[1])
    avg1.append(out[2])
    std1.append(out[3])
    if x0 == 0. and out[0] > 0.5:
        # just passed the iso = 0.5 mark while
        # increasing box size
        x0 = xrho[-1]
        x1 = xrho[-2]

# x, xaxis = xedge, "edges"
x, xaxis = xrho, "rho"
x_c = (x0+x1)/2.

# Blue is iso, cyan is nematic
first = 0
last = len(edges)
slic = slice(first,last)
plt.errorbar(x[slic], avg0[slic], yerr=std0[slic], fmt='-bo', ecolor='b', label=r'$y_0$')
plt.errorbar(x[slic], avg1[slic], yerr=std1[slic], fmt='-cd', ecolor='c', label=r'$y_1$')
ax = plt.gca()
ylims = ax.get_ylim()
marg = ax.margins() 
print ylims, marg
# plt.plot([x_c, x_c], [ylims[0],ylims[1]], color="k", linestyle='-') 
plt.axvline(x_c,color="k")
plt.gcf().set_size_inches(6,4)
plt.rc('axes', labelsize=14)
plt.xlabel(r'$\rho^*$')
plt.ylabel(r'$y$')
plt.minorticks_on()
lentrn = str(len(trnlist))

imgname = "fnn-output.eps"
plt.legend(fontsize='large')
plt.gcf().set_size_inches(6,4)
# plt.gcf().savefig(imgname);


# In[ ]:


sess.close()

