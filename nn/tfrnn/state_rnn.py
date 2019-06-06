import tensorflow as tf
import random
import sys, getopt
import time
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.contrib import rnn

global_tstart = time.time()

def stdout(s):
    sys.stdout.write(str(s)+'\n')

rundir = "/home/walterms/project/walterms/mcmd/nn/data/"
trndir = rundir+"train/"
testdir = rundir+"test/"
bThetas = False
nrod = 784 
nlabel = 10
batchsize = 200
nEpoch = 1
nRound = 1
ckptfile = "default_b"+str(batchsize)+".ckpt"
bCkptRestore = False
bSaveCkpt = False
runname = "mnist"

bEarlyStop = False

nchannel = 3
if bThetas: nchannel = 1

eta = 1e-3
nHidden = 512
nLayer = 1
keep_prob = 0.8

epochEval = int(10**(np.log10(nEpoch)//1 - 1))
if epochEval<1: epochEval=1

nblPerTrnFile = 10000 # 4000

arglen = len(sys.argv[1:])
arglist = sys.argv[1:]

try:
    opts, args = getopt.getopt(arglist,"e:b:r:",["epochEval=","bEarlyStop=","runname=","eta=","keepprob=","bSummaries=","ckpt=","bSaveCkpt=","bCkptRestore="])
except:
    stdout("Error in opt retrival...")
    stdout("seq_rnn")
    stdout("  -e num epoch")
    stdout("  -s sequence length")
    stdout("  -b batch size")
    stdout("  -o,--output output file tag")
    stdout("  --bSummaries record summaries (boolean)")
    stdout("  --ckpt checkpoint file")
    stdout("  --bCkptRestore (boolean)")
    sys.exit(2)

for opt, arg in opts:
    if opt == "-e":
        nEpoch = int(arg)
    elif opt == "--eta":
        eta = float(arg)
    elif opt == "--keepprob":
        keep_prob = float(arg)
    elif opt == "-b":
        batchsize = int(arg)
    elif opt == "--runname":
        runname = arg
    elif opt == "-r":
        nRound = int(arg)
    elif opt == "--epochEval":
        epochEval = int(arg)
    elif opt in ("--stepsize"):
        stepsize = int(arg)
    elif opt == "--bSummaries":
        if arg in ("False","false","0"):
            bSummaries = False
        elif arg in ("True","true","1"):
            bSummaries = True
        else:
            stdout("Fromat bSummaries properly")
            sys.exit(2)
    elif opt == "--bSaveCkpt":
        if arg in ("False","false","0"):
            bSaveCkpt = False
        elif arg in ("True","true","1"):
            bSaveCkpt = True
    elif opt == "--bEarlyStop":
        if arg in ("False","false","0"):
            bEarlyStop = False
        elif arg in ("True","true","1"):
            bEarlyStop = True
    elif opt == "--bCkptRestore":
        if arg in ("False","false","0"):
            bCkptRestore = False
        elif arg in ("True","true","1"):
            bCkptRestore = True
        else:
            stdout("Fromat bCkptRestore properly")
            sys.exit(2)
    elif opt == "--ckpt":
        ckptfile = arg

if nEpoch==-1 or batchsize==-1:
    stdout("Pass nEpoch and batchsize arguments")
    sys.exit(2)

stdout("batchsize "+str(batchsize))
stdout("nHidden "+str(nHidden))
stdout("nLayer "+str(nLayer))
stdout("keep_prob "+str(keep_prob))

trnlist = ["X","T","D","U"]
testlist = ["X","T","D","U"]

trnlist = ["mnist"]
testlist = ["mnist"]

'''
noise = 0.00
n_appnd = "_"+str(noise)
n_appnd = ""
jams = ["tjam", "xjam", "ujam", "ljam", "djam"]

for j in jams:
    trnlist.append(j+n_appnd)
    testlist.append(j)
'''
trnlist = [trndir+x for x in trnlist]
testlist = [testdir+x for x in testlist]

ckptdir = "/home/walterms/project/walterms/mcmd/nn/tfrnn/ckpts/state/"
ckptfile = ckptdir+ckptfile


def gen_labeled_set(flist,nblPerFile,nskip=0):
    img_set = []
    shufIdx = [p for p in range(nrod)]
    random.shuffle(shufIdx)
    for f in flist:
        stdout("Processing " + f + " for img set")
        thetas = []
        xs = []
        ys = []
        fin = open(f, 'r')
        nbl = 0
        skipped = 0
        for line in fin.readlines():
            if skipped < nskip:
                skipped+=1
                continue
            if line == "\n": continue
            if line.startswith("label"):
                # Done a block
                lblvec = np.zeros((nlabel))
                lbl = int(float(line.split()[1]))
                lblvec[lbl] = 1.
                
                # Insert data as triplets
                channels = [xs,ys,thetas]
                prep_data = []
                for ch in channels:
                    prep_data.append(np.asarray(ch))
                formatted_data = np.stack(prep_data)
                np.random.shuffle(formatted_data)
                img_set.append([formatted_data, lblvec])
                    
                thetas = []
                xs = []
                ys = []
                nbl+=1
                if nbl == nblPerTrnFile: 
                    break
                continue
            spt = [float(x) for x in line.split()]
            try:
                xs.append(spt[0])
                ys.append(spt[1])
                thetas.append(spt[2])
            except:
                stdout("fail")
                stdout(str(nbl))
                stdout(line)

        fin.close()
    return np.asarray(img_set)
    
###################
#       RNN       # 
###################

stdout("Constructing State RNN graph")

sizedict = {"nchannel": nchannel,
            "batchsize": batchsize,
            "nHidden": nHidden,
            "nLayer": nLayer,
            "nlabel": nlabel}

X = tf.placeholder("float", [None, nchannel, nrod])
Y = tf.placeholder("float", [None, nlabel])

weights = {
    'h1': tf.Variable(tf.random_normal([nHidden, nHidden],stddev=0.1, dtype=tf.float32), name='W1'),
    'out': tf.Variable(tf.random_normal([nHidden, nlabel],stddev=0.1, dtype=tf.float32), name='W'),
    
}
biases = {
    'b1': tf.Variable(tf.random_normal([nHidden],stddev=0.1, dtype=tf.float32), name='b1'),
    'out': tf.Variable(tf.random_normal([nlabel],stddev=0.1, dtype=tf.float32), name='bias')
}


# Define an lstm cell with tensorflow
def lstm_cell(nUnits,dropout=True):
    # Dropout params
    if dropout:
        return rnn.DropoutWrapper(rnn.BasicLSTMCell(nUnits),state_keep_prob=keep_prob)
    else:
        return rnn.BasicLSTMCell(nUnits)

def RNN(x, weights, biases):
    # Build high lvl rnn for time series
    x = tf.unstack(x, nchannel, 1)
    #cell = lstm_cell(nHidden)
    stack = rnn.MultiRNNCell([lstm_cell(nHidden) for _ in range(nLayer)])

    # Get lstm cell output
    outputs, states = rnn.static_rnn(stack, x, dtype=tf.float32)
    layer_trans = tf.add(tf.matmul(outputs[-1], weights['h1']), biases['b1'])
    layer_trans = tf.nn.sigmoid(layer_trans)
    # Linear activation, using rnn inner loop last output
    return tf.matmul(layer_trans, weights['out']) + biases['out']


logits = RNN(X,weights,biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
beta = 0.000
loss = tf.nn.softmax_cross_entropy_with_logits(\
    logits=logits, labels=Y)\
    + (tf.nn.l2_loss(weights['h1'])\
    + tf.nn.l2_loss(weights['out'])\
    + tf.nn.l2_loss(biases['b1'])\
    + tf.nn.l2_loss(biases['out']))*beta
loss_op = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(loss_op)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
stdout("Finished Graph")

##############
#  TRAINING  #
##############
# Compile training and test sets
stdout("Compiling training set")
trn_data = gen_labeled_set(trnlist,nblPerTrnFile)
stdout("Done. "+str(len(trn_data))+" images")

# Get proper number of test imgs
nblPerTestFile = (500//batchsize + bool(500%batchsize))*batchsize
stdout("Compiling test set")
nblTest = 10000
test_data = gen_labeled_set(testlist,nblTest)
stdout("Done. "+str(len(test_data))+" images")

# Saver for checkpoints
saver = tf.train.Saver()

batchesPerEpoch = len(trn_data)//batchsize
ntestbatches = len(test_data)//batchsize

xtest = np.asarray([t[0] for t in test_data])
ytest = np.asarray([t[1] for t in test_data])

k = 5
UPs = [0 for kk in range(k)]
Ps = [0. for kk in range(k)]
losses = np.zeros(shape=(k,2)) # train and test losses
# For now let's eval each epoch

with tf.Session() as sess:

    # Record file
    trnst = ""
    testst = ""
    for t in trnlist:
        trnst += t+ " "
    for t in testlist:
        testst += t+" "
    fin = open("/home/walterms/project/walterms/mcmd/nn/tfrnn/records/"+runname,'w')
    fin.write("eta "+str(eta))
    fin.write("\nnHidden "+str(nHidden))
    fin.write("\nnLayer "+str(nLayer))
    fin.write("\nbThetas "+str(bThetas))
    fin.write("\ntrnlist "+trnst)
    fin.write("\ntestlist "+testst)
    fin.write("\nbatchsize "+str(batchsize))
    fin.write("\nkeepprob "+str(keep_prob))
    fin.write("\nnEpoch "+str(nEpoch))
    fin.write("\nentries [epoch trncost trnacc testcost testacc]\n")

    for r in range(nRound):
        
        stdout("Round "+str(r)+" of "+str(nRound))
        fin.write("\n")

        # Checkpoint file
        if bCkptRestore:
            stdout("Restoring from "+ckptfile)
            saver.restore(sess, ckptfile)
            stdout("Model restored")
        else:
            stdout("Initializing variables.")
            sess.run(tf.global_variables_initializer())

        for e in range(nEpoch):
            nmax = 60000
            # nblPerTrnFile should be 10 000 for mnist
            linestart = nblPerTrnFile*(e%6)*(28*28 + 2)

            trn_data = gen_labeled_set(trnlist,nblPerTrnFile,nskip=linestart)
            avg_loss = 0.
            batchIdx = [bb for bb in range(batchesPerEpoch)]
            random.shuffle(batchIdx)
            trn_acc = 0.
            for b in batchIdx:
                ib = b*batchsize
                # Pepare a batch
                yin = np.asarray([trn_data[ib+iib][1] for iib in xrange(batchsize)])
                xin = np.asarray([trn_data[ib+iib][0] for iib in xrange(batchsize)])

                _,l,tacc = sess.run([optimizer, loss_op, accuracy], feed_dict={X: xin, Y: yin})

                trn_acc += tacc / batchesPerEpoch
                avg_loss += l / batchesPerEpoch
                

            if e % epochEval == 0:
                # Eval on test set
                avg_testloss = 0.
                testacc = 0.
                for tb in range(ntestbatches):
                    ib = tb*batchsize
                    xin = xtest[ib:ib+batchsize]
                    yin = ytest[ib:ib+batchsize]
                    acc, youts, testloss = sess.run([accuracy,prediction,loss_op],feed_dict={
                        X: xin, Y: yin})
                    testacc += acc/float(ntestbatches)
                    avg_testloss += testloss/float(ntestbatches)
                print('Epoch %d\nTest:\taccuracy %3.6f\tloss %3.6f\nTrain:\taccuracy %2.6f\tloss %2.6f'
                     % (e,testacc,avg_testloss,trn_acc,avg_loss))

                fin.write('%d %g %g %g %g\n'%(e,avg_loss,trn_acc,avg_testloss,testacc))

                if bEarlyStop:
                    # Let's use UP5 as early stopping criterion with k=5
                    loss0 = losses[0]
                    losses[e%k][0] = avg_loss
                    losses[e%k][1] = avg_testloss
                    if e%k == k-1:
                        stdout("Evaluating strip")
                        minEtr = np.min(losses[:,0])
                        for ik in range(k-1):
                            Ps[ik] = Ps[ik+1]
                            UPs[ik] = UPs[ik+1]
                        Ps[k-1] = 1000.*(np.sum(losses[:,0]) / (k*minEtr) - 1)
                        UPs[k-1] = losses[k-1,1] > loss0[1]

                        stdout(UPs)


                    if sum(UPs) == k:
                        # Time to stop
                        stdout("Early stopping criterion satisfied. Done training.")
                        break

        
        if bSaveCkpt:
            # Saving checkpoint
            stdout("Saving checkpoint to "+ckptfile)
            save_path = saver.save(sess, ckptfile)
            stdout("Saved checkpoint")

stdout("Done")
global_tend = time.time()
stdout("Total time: "+str(global_tend-global_tstart))
