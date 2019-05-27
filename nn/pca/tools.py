import matplotlib.pyplot as plt
import numpy as np

cos = np.cos
sin = np.sin
twopi = np.pi*2.
pi = np.pi

def myrotate(th):
    # Add pi/2 to make it reference the x-axis
    # on range [0,2pi]
#     th = (th+pi/2.) if th<(3.*pi/2.) else (th-3.*pi/2.)
#     if th > pi: th -= pi
    th = th + pi/2.
    if th > twopi: th-= twopi
    return th


def dist_from(ref,rs):
    # ref is 1x2 ndarray
    # rs is list of vectors to compare, nx2
    return np.linalg.norm((ref-rs),axis=1)

def nematicdirector(Q):
    # Given Q matrix return order parameter and angle
    w,v = np.linalg.eig(Q)
    idx = np.argmax(w)
    Lam = w[idx] # Equiv to sqrt(S*S+T*T)
    nu = v[:,idx]
    alpha = np.angle(np.complex(nu[0,0],nu[1,0]))
    if alpha < 0: alpha = twopi+alpha
    return Lam, alpha


def quad_features(pca_data):
    # pca_data should be the nbr vector after the pca transformation
    # Let's have this function return feature 1 as f_1^2 +f_2^2
    # and feature 2 as f_3^2 + f_4^2
    new_pca_data = np.zeros(shape=(pca_data.shape[0], pca_data.shape[1]-2),dtype=float)
    new_pca_data[:,0] = pca_data[:,0]
    new_pca_data[:,1] = np.sqrt(np.square(pca_data[:,1]) + np.square(pca_data[:,2]))
    new_pca_data[:,2] = np.sqrt(np.square(pca_data[:,3]) + np.square(pca_data[:,4]))
    new_pca_data[:,3:] = pca_data[:,5:]
    return new_pca_data    
    

def gen_probes(nx,edge,do_shift=True):
    nprobe = nx**2
    probes = np.zeros((nprobe,2))
    dx = edge/nx
    dy = edge/nx
    halfdx = dx/2.
    shift = 0
    if do_shift:
        shift = -edge/2.
    iprobe = 0
    for iy in range(nx):
        for ix in range(nx):
            # Start from top left of image and read right
            probes[iprobe] = [dx*ix + halfdx + shift, -shift - dy*iy - halfdx]
            iprobe += 1
            
    return probes


def norm_angles(thetas,alpha):
    '''
    Given 1D array of angles, express the angles relative to alpha
    thetas and alpha should come in range [0,2pi]
    '''
    th0 = alpha
    for ith, th in enumerate(thetas):
        dth = th - th0
        if dth > pi/2.:
            if dth > 3*pi/2.:
                th -= twopi
            else:
                th -= pi
        if dth < -pi/2.:
            if dth < -3*pi/2.:
                th += twopi
            else:
                th += pi
        thetas[ith] = th
        th0 = th
    return thetas


def winding_angles(thetas,alpha):
    '''
    Given 1D array of angles, return their angles relative to alpha
    such that the polarity is closest to that of the angle previous to it
    in the array
    I think it boils down to deciding on a direction that things are spinning
    After that we can make a better adjustment of the polarity conversion
    region, say [(+/-) pi/4, (-/+) 3pi/4]
    There are two ideas that come to mind: a threshold method or averaging method
    for determining when to decide direction
    '''
    thresh = 0.2
    pol = 0
    th0 = alpha
    dalpha = 0
    for ith, th in enumerate(thetas):
        dth = th - th0
#         print th, dth
        if dth > pi/2 + pol*pi/4:
            if dth > 3*pi/2 + pol*pi/4:
                th -= twopi
            else:
                th -= pi
        if dth < -pi/2  + pol*pi/4:
            if dth < -3*pi/2 + pol*pi/4:
                th += twopi
            else:
                th += pi
        dalpha = th - alpha
        if (abs(dalpha) > thresh) and (pol==0):
            pol = np.sign(dalpha)
        
#         print th, dalpha, pol
        thetas[ith] = th
        th0 = th
    return thetas - alpha


def polar_cluster_sort(rods,alpha,nndist=0.):
    '''
    rod x,y coordinates must be relative to their probe center
    angular coordinates are in range [0,2pi]
    Given your application, they will have their 4th column as a 
    variable vector
    Instead of strict polar sort, the next rod is actually a nearest
    neighbour in the winding direction
    We can achieve this by first checking to see if there are any
    neighbours in the nndist along the winding direction
    We can sort this in place. Let's add a new column that we
    update with nn distances to the current rod, and we only check 
    rods that are upcoming
    Notice, rods are already sorted based on distance to center
    
    Returns rods with final dimension being th_jalpha
    '''
    nrod = len(rods)
    idxs = np.arange(nrod).reshape(nrod,1)
    rods = np.append(idxs,rods,axis=1)
    rods2 = rods.copy()
    phi = np.arctan2(rods[:,2],rods[:,1])
    phi = np.where(phi < 0., phi+twopi, phi)
    phi_jalpha = phi - alpha
    phi_jalpha = np.where(phi_jalpha < 0., phi_jalpha+twopi, phi_jalpha)
    
    rods[:,-1] = phi_jalpha
    rods = rods[rods[:,-1].argsort()]
    rods3 = rods.copy()
    # rods2 is be sorted by dist from center
    # rods is sorted by phi_jalpha
    # rods3 is also sorted by phi_jalpha
    checked = []
    rod3cnt = 0
    for rod in rods:
        start = int(rod[0])
        if start not in checked:
            # Add to rods3
            checked.append(start)
            rods3[rod3cnt] = rod
            rod3cnt += 1
#         end = min(start+5,nrod)
        cent = np.asarray([rod[1],rod[2]])
        dists = dist_from(cent,rods2[:,1:3])
        rods2[:,-1] = dists
        rods2 = rods2[rods2[:,-1].argsort()]
        rodnbrs = rods2[1:6]
        for nbr in rodnbrs:
            if int(nbr[0]) in checked:
                continue
            # Add nbr rods to rods3 if needed
            # See if it's close to our subject rod
            # and if it is further along phi
            dphi = rods[int(nbr[0]), -1] - rods[start,-1]
            if (nbr[-1] < nndist) and (dphi > 0.):
                checked.append(int(nbr[0]))
                rods3[rod3cnt] = nbr
                rod3cnt+=1
    
    if abs(sum(rods[:,1]) - sum(rods3[:,1])) > 0.00001:
        print "wuhoh"
    
    return rods3[:,1:]


def get_lat_nbrs(block,n_nbr,edge,nx,probes,use_bulk=False,\
                 method="random",ret_nbrs=False,sparse_bulk_factor=1,\
                 use_xyth=False):
                 
    '''
    Return (xprobe, n_nbr) array of neighbours
    I use xprobe because if use_bulk is true then
    It's some indeterminant amount
    use_bulk should maybe not be true 
    
    method can be one of: random, radial, polar, angular
    random and radial are self-explanatory
    polar is sorted by rod center location order while circling
    around starting with the nem director
    angular is ordered by those closest to the nem director
    
    return 2D array "features"
    '''
    
    # Create probes
    nprobe = nx**2
    
    block2 = block.copy()
    nrod = len(block)
    z = np.zeros((nrod,1))
    block2 = np.append(block2,z,axis=1)
    
    features = np.zeros(shape=(nprobe,n_nbr)) if not (use_xyth) else np.zeros(shape=(nprobe,n_nbr*3))
    nbrs_full = np.empty(shape=(nprobe,n_nbr,4))
    nbrs = np.empty(shape=(n_nbr,4))
    alphas = np.zeros(shape=(nprobe))
    
    nnbr_ft = 3

    for i,prob in enumerate(probes):
        cent = np.asarray([prob[0],prob[1]])
        dists = dist_from(cent,block2[:,:2])
        block2[:,-1] = dists
        block2 = block2[block2[:,-1].argsort()]
        
        nbrs = np.copy(block2[:n_nbr])
        
        # Convert coordinates, relative to center of mass
#         com_x, com_y = np.mean(nbrs[:,0]), np.mean(nbrs[:,1])
        com_x, com_y = prob[0], prob[1]
        nbrs[:,0] -= com_x
        nbrs[:,1] -= com_y

        # Convert angles to be relative to nem director
        # Angles are originally [0,2pi]
        th_j = nbrs[:,2]
        S,T = np.mean(cos(2.*th_j)), np.mean(sin(2.*th_j))
        Q = np.matrix([[S,T],[T,-S]])
        _, alpha = nematicdirector(Q) # alpha is range [0,2pi]
        alphas[i] = alpha
        
        th_jalpha = th_j - alpha # [-2pi,2pi]

        if method == "random":
            np.random.shuffle(nbrs)
            th_jalpha = th_j - alpha
            if use_xyth:
                nbrs[:,-1] = cos(2.*th_jalpha)
                features[i] = nbrs[:,:nnbr_ft].flatten()
            else:
                features[i,:] = cos(2.*th_jalpha)
                nbrs[:,-1] = features[i,:]
            
        if method == "radial":
            if use_xyth:
                nbrs[:,-1] = cos(2.*th_jalpha)
                features[i] = nbrs[:,:nnbr_ft].flatten()
            else:
                features[i,:] = cos(2.*th_jalpha)
                nbrs[:,-1] = features[i,:]
            
        if method == "angular":
            # sort by cos(2*th_jalpha)
            # we use argsort()[::-1] so that most aligned appear first
            # Normalize rod angles first
            nbrs = polar_cluster_sort(nbrs,alpha,nndist=0.4)
            nbrs[:,-1] = winding_angles(nbrs[:,2],alpha) / pi
            nbrs[:,-1] = np.square(nbrs[:,-1])
#             nbrs[:,-1] = cos(2.*th_jalpha)
#             nbrs[:,-1] = (norm_angles(nbrs[:,2],alpha))
            nbrs = nbrs[nbrs[:,-1].argsort()]
            if use_xyth:
                features[i] = nbrs[:,:nnbr_ft].flatten()
            else:
                features[i,:] = nbrs[:,-1]
            
        if method == "polar":
            nbrs = polar_cluster_sort(nbrs,alpha,nndist=0.4)
#             features[i,:] = cos(2.*th_jalpha)
#             features[i,:] = (norm_angles(nbrs[:,2],alpha) - alpha) / pi
            nbrs[:,-1] = cos(2.* winding_angles(nbrs[:,2],alpha) )
            if use_xyth:
                features[i] = nbrs[:,:nnbr_ft].flatten()
            else:
                features[i,:] = nbrs[:,-1]
            
        # Return nbrs to original coords
        nbrs[:,0] += com_x
        nbrs[:,1] += com_y
        
        nbrs_full[i] = nbrs
        
    if ret_nbrs:
        return features, nbrs_full, alphas
    else:
        return features

    
    
def plotLine(x1,y1,x2,y2,c='b',ax=None,lw=0.4,alpha=1.0):
    if ax: # given axis handle
        ax.plot([x1, x2], [y1, y2], color=c, linestyle='-', linewidth=lw, alpha=alpha);
    else:
        plt.gca().plot([x1, x2], [y1, y2], color=c, linestyle='-', linewidth=lw, alpha=alpha);
        
def plotrods(rods,myax,halfL=0.5,hotrods=[],col='k',lw=0.4,alpha=1.0,add_crosses=False):
    for r in rods:
        th = r[2]
        x1 = r[0] - halfL*cos(th)
        x2 = r[0] + halfL*cos(th)
        y1 = r[1] - halfL*sin(th)
        y2 = r[1] + halfL*sin(th)
        plotLine(x1,y1,x2,y2,c=col,lw=lw,ax=myax,alpha=alpha)
        if add_crosses:
            myax.plot(r[0],r[1],"+",markersize=20,color="grey",linewidth=5)
        
    if len(hotrods)>0:
        for r in hotrods:
            th = r[2]
            x1 = r[0] - halfL*cos(th)
            x2 = r[0] + halfL*cos(th)
            y1 = r[1] - halfL*sin(th)
            y2 = r[1] + halfL*sin(th)
            plotLine(x1,y1,x2,y2,c='r',lw=1.6,ax=myax,alpha=alpha)
            

            
def get_nbrs(block,n_nbr,edge,use_bulk=False,method="random",ret_nbrs=False,sparse_bulk_factor=1):
    '''
    Return (xrod, n_nbr) array of neighbours
    I use xrod because if use_bulk is true then
    It's some indeterminant amount
    use_bulk should maybe not be true 
    
    method can be one of: random, radial, polar, angular
    random and radial are self-explanatory
    polar is sorted by rod center location order while circling
    around starting with the nem director
    angular is ordered by those closest to the nem director
    
    return 2D array "features"
    '''

    bulk = block.copy()
    bulk = bulk[np.where(bulk[:,0] > -0.25*edge)]
    bulk = bulk[np.where(bulk[:,0] < 0.25*edge)]
    bulk = bulk[np.where(bulk[:,1] > -0.25*edge)]
    bulk = bulk[np.where(bulk[:,1] < 0.25*edge)]
    bulk = bulk[::sparse_bulk_factor]

    perim = block.copy()
    perimidx = np.where(perim[:,0] < -0.25*edge)
    perimidx = np.append(perimidx,np.where(perim[:,0] > 0.25*edge))
    perimidx = np.append(perimidx,np.where(perim[:,1] < -0.25*edge))
    perimidx = np.append(perimidx,np.where(perim[:,1] > 0.25*edge))
    perimidx = np.unique(perimidx) # remove double counts
    perim = perim[perimidx]
    
    probes = np.append(perim,bulk,axis=0)
    
    # This is mainly just for viewing the rods in gen_nbrfiles
    if sparse_bulk_factor == 1:
        probes = block.copy()

    block2 = block.copy()
    nrod = len(block)
    nprobe = len(probes)
    z = np.zeros((nrod,1))
    block2 = np.append(block2,z,axis=1)
    
    features = np.zeros(shape=(nprobe,n_nbr))
    nbrs_full = np.empty(shape=(nprobe,n_nbr,4))
    nbrs = np.empty(shape=(n_nbr,4))
    alphas = np.zeros(shape=(nprobe))

    for i,rod in enumerate(probes):
        cent = np.asarray([rod[0],rod[1]])
        dists = dist_from(cent,block2[:,:2])
        block2[:,-1] = dists
        block2 = block2[block2[:,-1].argsort()]
        
        nbrs = np.copy(block2[1:1+n_nbr])
        # Convert coordinates, relative to center of mass
        cx, cy = np.mean(nbrs[:,0]), np.mean(nbrs[:,1])
        nbrs[:,0] -= cx
        nbrs[:,1] -= cy

        # Convert angles to be relative to nem director
        th_j = nbrs[:,2]
        S,T = np.mean(cos(2.*th_j)), np.mean(sin(2.*th_j))
        Q = np.matrix([[S,T],[T,-S]])
        _, alpha = nematicdirector(Q)
        
        # Try using rod probe as alpha
#         alpha = rod[2]

        th_jalpha = th_j - alpha
        alphas[i] = rod[0]

        if method == "random":
            np.random.shuffle(nbrs)
            th_jalpha = th_j - alpha
            features[i,:] = cos(2.*th_jalpha)
            nbrs[:,-1] = features[i,:]
            
        if method == "radial":
            features[i,:] = cos(2.*th_jalpha)
            nbrs[:,-1] = features[i,:]            
            
        if method == "angular":
            nbrs[:,-1] = cos(2.*th_jalpha)
            # sort by cos(2*th_jalpha)
            # we use argsort()[::-1] so that most aligned appear first
            nbrs = nbrs[nbrs[:,-1].argsort()[::-1]]
            features[i,:] = nbrs[:,-1]
            nbrs[:,-1] = features[i,:]
            
        if method == "polar":
            # get polar coordinates
            # arctan2 returns [-pi,pi]
            phi = np.arctan2(nbrs[:,1],nbrs[:,0])
            phi = np.where(phi < 0., phi+twopi, phi)
            
            phi_jalpha = phi - alpha
            phi_jalpha = np.where(phi_jalpha < 0., phi_jalpha+twopi, phi_jalpha)
            nbrs[:,-1] = phi_jalpha
            nbrs = nbrs[nbrs[:,-1].argsort()]
            th_jalpha = nbrs[:,2] - alpha
            features[i,:] = cos(2.*th_jalpha)
            nbrs[:,-1] = features[i,:]
            
        # Return nbrs to original coords
        nbrs[:,0] += cx
        nbrs[:,1] += cy
        
        nbrs_full[i] = nbrs
        
    if ret_nbrs:
        return features, nbrs_full, alphas
    else:
        return features
    
