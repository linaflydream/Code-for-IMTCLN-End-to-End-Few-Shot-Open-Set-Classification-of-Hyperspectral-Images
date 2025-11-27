import numpy as np
import random
class SLSLDE(object):
    def location(self,m1,m2):
        Location = np.zeros([m1,m2,2])
        
        Location[:,:,0] = np.tile(np.linspace(0,m1-1,m1,dtype='int').reshape([m1,1]),[1,m2])
        Location[:,:,1] = np.tile(np.linspace(0,m2-1,m2,dtype='int').reshape([1,m2]),[m1,1])

        Location = Location.reshape([m1*m2,2])
        return Location
    def get_linyu(self,x,s):
        [m,n,D] = x.shape
        y = np.zeros([(m-s+1)*(n-s+1),s,s,D])
        e=int(0)
        k=int((s-1)/2)
        start = int(((s+1)/2)-1)
        xendm = int(m-(s-1)/2)
        xendn = int(n-(s-1)/2)
        for i in range(start,xendm):
            for j in range(start,xendn):
                y[e] = x[i-k:i+k+1,j-k:j+k+1,:]
                e = e+1
        return y
        
    def kuozhan(self,x,s):
        [a,b,c] = x.shape
        k = int(s-1)
        ks = int(k/2)
        y = np.zeros([a+k,b+k,c])
        y[ks:-ks,ks:-ks,:] = x

        y[ks:-ks,0:ks,:] = np.fliplr(x[0:a,1:ks+1,:])
        y[0:ks,ks:-ks,:] = np.flipud(x[1:ks+1,0:b,:])
        y[ks:-ks,-ks:,:] = np.fliplr(x[0:a,-ks-1:-1,:])
        y[-ks:,ks:-ks,:] = np.flipud(x[-ks-1:-1,0:b,:])
    
        y[0:ks,0:ks,:] = np.rot90(x[1:ks+1,1:ks+1,:],2)
        y[0:ks,-ks:,:] = np.rot90(x[1:ks+1,-ks-1:-1,:],2)
        y[-ks:,0:ks,:] = np.rot90(x[-ks-1:-1,1:ks+1,:],2)
        y[-ks:,-ks:,:] = np.rot90(x[-ks-1:-1,-ks-1:-1,:],2)
        return y
    def WMF(self,X,S):
        [m1,m2,D] = X.shape
        N = m1*m2
        X_SS = self.get_linyu(self.kuozhan(X,S),S)
        xs = X.reshape([N,1,1,D])
        vk = np.exp(-0.2*np.sum(np.square(xs-X_SS),3))
        vk = vk.reshape([N,S,S,1])
        xf = np.sum(vk*X_SS,(1,2))/np.sum(vk,(1,2))
        xf = xf.reshape([N,D])
        return xf
    
    def generate_data(self,X,Location,label,m1,m2,S):
        [N,D] = X.shape
        Location = np.sin(np.true_divide(Location,m1))
        X_L = np.zeros([N,D+2])
        X_L[:,0:D] = X
        X_L[:,-2:] = Location
        X_L = X_L.reshape([m1,m2,D+2]) 
        X_SS_L = self.get_linyu(self.kuozhan(X_L,S),S)
        X_SS_short_L = X_SS_L[np.where(label!=0)[0],:,:,:]
        return X_SS_short_L
    def generate_data2(self,X,Location,label,m1,m2,S):
        [N,D] = X.shape
        num_class=label.max()
        Location = np.sin(np.true_divide(Location,m1))
        X_L = np.zeros([N,D+2])
        X_L[:,0:D] = X
        X_L[:,-2:] = Location
        X_L = X_L.reshape([m1,m2,D+2])
        X_SS_L = self.get_linyu(self.kuozhan(X_L,S),S)
        X_SS_short_L = X_SS_L[np.where(np.logical_and(label != 0, label !=num_class))[0],:,:,:]
        # X_SS_short_L = X_SS_L[np.where(label != 0)[0], :, :, :]
        return X_SS_short_L
    def find_Ni_labels(self,Ni,label,Class):
        indices = []
        label = label - 1
        for i in range(Class):
            ind  = np.where(label == i)[0]
            L = len(ind)
            p = list(range(L))
            random.shuffle(p)
            if Ni <= L:
                N = Ni
            elif Ni > L:
                N = round(0.5*L)
            indices = np.append(indices,ind[p[0:N]])
        return indices.astype('int')
    
    def generate_label(self,label,num_class):
        N = label.shape[0]
        label = label - 1
        y = np.zeros([N,num_class])
        for i in range(N):
            y[i,label[i]] = 1
        return y
























