import tensorflow as tf
import random
import numpy as np
import time
import sys, getopt
from tensorflow.contrib import rnn

global_tstart = time.time()

def stdout(s):
    sys.stdout.write(str(s)+'\n')

nrod = 400
NLBL = 6
batchsize = -1
seq_len = -1 
nEpoch = -1
eta = 1e-2
nInput = nrod
nHidden = 32
nDense = 32
subnlayer = 1
seqnlayer = 1

infile = ""
outfile = ""
ckptfile = None

arglen = len(sys.argv[1:])
arglist = sys.argv[1:]
try:
    opts, args = getopt.getopt(arglist,"s:b:i:o:",["infile=","outfile=","ckpt="])
except:
    stdout("Error in opt retrival...")
    stdout("seq_rnn")
    stdout("  -s sequence length")
    stdout("  -b batch size")
    stdout("  -i,--infile input file")
    stdout("  -o,--outfile output file")
    stdout("  --ckpt checkpoint file")
    sys.exit(2)

for opt, arg in opts:
    if opt == "-s":
        seq_len = int(arg)
    elif opt == "-b":
        batchsize = int(arg)
    elif opt in ("-i","--infile"):
        infile = arg
    elif opt in ("-o","--outfile"):
        outfile = arg
    elif opt == "--ckpt":
        ckptfile = arg

ckptdir = "/home/walterms/project/walterms/mcmd/nn/tfrnn/ckpts/"


###################
#  PREPARE INPUT  #
#       DATA      #
###################

# Count Nbl
Nbl = 0
fin = open(infile,'r')
for line in fin.readlines():
    if line == "\n":
        Nbl+=1
fin.close()
Nbl = Nbl - (Nbl%batchsize)
stdout("Effective Nbl in input: "+str(Nbl))

features = ["x","y","th"]
# features = ["x","y","ft1","ft2"]
featdict = {}
for ft in features:
    featdict.update({ft:[]})

nchannel = len(features)

def gen_seq_set(f):
    stdout("Processing input file "+infile)
    sortIdx = np.arange(nrod,dtype=int)
    IDs = []
    fin = open(f, 'r')
    seqset = []
    for line in fin.readlines():
        if line == "\n":
            # Done a block
            # Sort based on rod indices
            sortIdx = np.argsort(IDs)
            
            # Insert data as triplets
            channels = []
            for ft in features:
                channels.append(featdict[ft])
            prep_data = []
            for ch in channels:
                prep_data.append(np.asarray(ch)[sortIdx])
            formatted_data = np.stack(prep_data)
            seqset.append(formatted_data)
                
            for ft in features:
                featdict[ft] = []
            IDs = []
            continue
        spt = [float(x) for x in line.split()]
        featdict["x"].append(spt[0]-0.5)
        featdict["y"].append(spt[1]-0.5)
        th = spt[2]-0.5
        featdict["th"].append(th)
        
        IDs.append(int(spt[3]))

    fin.close()
    return np.asarray(seqset)
    
# Generate seq sets
stdout("Generating input data...")
in_seq = gen_seq_set(infile)[0:Nbl]
stdout("Done")
    

###################
#       RNN       # 
###################

def variable_summaries(var):
    #A ttach a lot of summaries to a Tensor (for TensorBoard visualization)
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


X = tf.placeholder("float", [None, seq_len, nchannel, nInput],name="X")
Y = tf.placeholder("float", [None, nchannel, nInput],name="Y")

with tf.name_scope('dense'):
    dense_weights = {"pre":tf.Variable(tf.random_normal([nHidden,nDense],
                stddev=0.1,dtype=tf.float32),name="pre_w")}
    for f in features:
        dense_weights.update({f:tf.Variable(tf.random_normal([nDense,nrod],
                stddev=0.1,dtype=tf.float32),name=f+"_w")})

    dense_biases = {"pre":tf.Variable(tf.random_normal([nDense],
                stddev=0.1,dtype=tf.float32),name="pre_b")}
    for f in features:
        dense_biases.update({f:tf.Variable(tf.random_normal([nrod],
                stddev=0.1,dtype=tf.float32),name=f+"_b")})
        
    for w in dense_weights:
        tf.summary.histogram(w+"_ws",dense_weights[w])
    for b in dense_biases:
        tf.summary.histogram(b+"_bs",dense_biases[b])


# Define an lstm cell with tensorflow
def lstm_cell(nUnits):
    return rnn.BasicLSTMCell(nUnits)

def seqRNN(x):

    x = tf.unstack(x,seq_len,1) # unstack along time dimension
    
    with tf.name_scope('subrnn'):
        with tf.variable_scope('subrnn'):
            # Subcell    
#             subcell = lstm_cell(nHidden)
            subcell = rnn.MultiRNNCell([lstm_cell(nHidden) for _ in range(subnlayer)])

            suboutputs = []
            substate = subcell.zero_state(batchsize,tf.float32)

            # Loop over the images in a sequence
            for x_img in x:
                x_ = tf.unstack(x_img,nchannel,1)
                # Returns multiple outputs I think of size [batchsize,nchannel,subcell.output_size]
                suboutput_img, substate = tf.nn.static_rnn(subcell,x_,dtype=tf.float32,initial_state=substate)
                # suboutput_img is a list of 3 outputs from each iteration on the img
                # suboutput_img[-1] is the last output, let's use that as input to the seqrnn
                suboutputs.append(suboutput_img[-1])

            tf.summary.histogram('substate',substate)

    with tf.name_scope('seqrnn'):
        with tf.variable_scope('seqrnn'):
            # Main cell
#             cell = lstm_cell(nHidden)
            cell = rnn.MultiRNNCell([lstm_cell(nHidden) for _ in range(seqnlayer)])

            outputs,state = tf.nn.static_rnn(cell,suboutputs,dtype=tf.float32)
            tf.summary.histogram('cellstate',state)


    # Dense output from seqrnn
    with tf.name_scope('dense'):
        dense_pre = tf.nn.elu(tf.add(tf.matmul(outputs[-1],dense_weights["pre"]),
                        dense_biases["pre"]),name="pre_out_activ")

        # Tensors for transforming output of main RNN unit into an img
        out_img_channels = []
        i = 0
        for ft in features:
            out_img_channels.append(tf.nn.tanh(tf.add(tf.matmul(
                dense_pre,dense_weights[ft]),dense_biases[ft]),name=str(ft)+"_out_activ"))

            tf.summary.histogram(str(ft)+"_out",out_img_channels[-1])
            i+=1
    
    return tf.stack(out_img_channels,axis=1)


# Outputs a list of tensors of size nrod representing the img
seq_img = seqRNN(X)

stdout("Created RNN")


###################
#   Feedforward   # 
###################

nSeq = Nbl-seq_len
nbatches = nSeq//batchsize
saver = tf.train.Saver()

stdout("Opening "+outfile+" for writing")
fout = open(outfile,'w')

stdout("Beginning Session")
if not ckptfile:
    stdout("Need checkpoint file! Exiting...")
    sys.exit(2)
with tf.Session() as sess:
    # Restore from checkpoint file
    ckptfile = ckptdir+ckptfile
    stdout("Restoring from "+ckptfile)
    saver.restore(sess, ckptfile)
    stdout("Model restored")

    tstart = time.time()
    start = 0
    for ib in range(nbatches):
        # Pepare a batch
        end = start+batchsize
        yin = np.asarray([in_seq[i_img+seq_len] for i_img in range(start,end)])
        xin = np.asarray([[in_seq[i_img+s] for s in range(seq_len)] \
                for i_img in range(start,end)])
        
        outimgbatch, = sess.run([seq_img], feed_dict={X:xin,Y:yin})
        outimgbatch = outimgbatch.transpose((0,2,1))

        for img in outimgbatch:
            for line in img:
                fout.write("%.6f %.6f %.6f\n"%(line[0],line[1],line[2]))
            fout.write("\n")

        start = end


    stdout("Done feeding infile")

fout.close()
global_tend = time.time()
stdout("Total time: "+str(global_tend-global_tstart))
