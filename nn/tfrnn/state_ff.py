if not ckptfile:
    stdout("Need checkpoint file! Exiting...")
    sys.exit(2)

# Compile unlabeled data
unlbl_data = []
maxUnlbl = -1
            
for f in unlbllist:
    stdout("Processing " + f + " as training data")
    thetas = []
    xs = []
    ys = []
    fin = open(f, 'r')
    nbl = 0
    for line in fin.readlines():
        if line == "\n":
            # Done a block
            if bThetas:
                thdata = None
                random.shuffle(thetas)
                thdata = (-0.5) + np.asarray(thetas)
                formatted_data = np.stack([thdata,thdata,thdata])
#                 thdata = thdata.reshape(1,nrod)
                trn_data.append([formatted_data, lbls])

            else:
                # Insert data as triplets
                channels = [xs,ys,thetas]
                prep_data = []
                for ch in channels:
                    prep_data.append((-0.5)+np.asarray(ch))
                formatted_data = np.stack(prep_data)
#                 formatted_data = np.stack([np.asarray(xs),np.asarray(ys),np.asarray(thetas)])
                np.random.shuffle(formatted_data)
                unlbl_data.append([formatted_data, lbls])
                
            thetas = []
            xs = []
            ys = []
            nbl+=1
            if nbl == maxUnlbl: 
                break
            continue
        spt = [float(x) for x in line.split()]
        xs.append(spt[0])
        ys.append(spt[1])
        thetas.append(spt[2])

    fin.close()
    
unlbl_data = np.asarray(unlbl_data)
    
stdout("Done compiling unlabeled set,"+len(unlbl_data)+"images")


# In[ ]:




with sess.as_default():
    assert tf.get_default_session() is sess
    # Pepare a batch
    xin = np.asarray([trn_data[ib+iib][0] for iib in xrange(batchsize)])

    _,l = sess.run([optimizer, loss_op], feed_dict={X: xin, Y: yin})
    # Run optimization op (backprop)
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
        print('epoch %d, test accuracy %g, testloss %g, avg_loss %g'
             % (e,testacc,avg_testloss,avg_loss))
        y_out_0.append(youts[0])

stdout("Done")


# In[ ]:


# Loss plot
# Mask on zeros
f, ax = plt.subplots(1,2)
for r in records[-1:]:
    x = np.argwhere(r.testlosses>0)
    ax0, = ax[0].plot(x,r.trnlosses[x],'b-',label='train loss')
    axtwin = ax[0].twinx()
    ax0t, = axtwin.plot(x,r.testlosses[x],'g:',label='test loss')
    l1 = plt.legend([ax0,ax0t],["train loss", "test loss"])
    ax[1].plot(x,r.trnacc[x],'b-',label="train")
    ax[1].plot(x,r.testacc[x],'g:',label="test")
    ax[1].legend()
    ax[0].set_yscale("log", nonposy='mask')
    axtwin.set_yscale("log", nonposy='mask')
    
plt.gcf().set_size_inches(14,5)

