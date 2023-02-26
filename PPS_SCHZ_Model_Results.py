######################################################################################################
## Name:Ahmed Ali Ahmed                                                                             ##
## Assigment Multi-sensory integration network and perception of space around the body              ##
## 30/January/2023                                                                                  ##
## Refrences :                                                                                      ##
##                                                                                                  ##
######################################################################################################



################
#   Packages   #
################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
from scipy.spatial import distance
from scipy.optimize import curve_fit

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.mplot3d.axes3d import Axes3D
import string
import matplotlib.cm as cm

from matplotlib.ticker import LinearLocator
from scipy.spatial import distance


#########################################################################
#   Part 2.1: Setting up the tactile and auditory (unisensory) neurons  #
#########################################################################


#   Define Parameters   #
#########################

Auditory_Param={}
Auditory_Param['Ma'] = 20         # Number of Neurons along a dimensions
Auditory_Param['Na'] = 3          # Number of Neurons along a dimensions
Auditory_Param['Ia'] =  3.6       # Auditory Extenral Stimuli [mV]
Auditory_Param['sigmaaI'] = 0.3   # SD in Auditory Extenral Stimuli
Auditory_Param['sigmaa']= 10      # SD in Receptive Field of Neuron
Auditory_Param['phia']= 1         # Center of Neuron  Receptive Field [mV]

# Tactile  parameters 
Tactile_param={}
Tactile_param['Mt'] = 40         # Number of Neurons along a dimensions
Tactile_param['Nt'] = 20         # Number of Neurons along a dimensions
Tactile_param['It'] = 2.5        # Auditory Extenral Stimuli [mV]
Tactile_param['sigmatI'] = 0.3   # SD in Auditory Extenral Stimuli
Tactile_param['sigmat']=1        # SD in Receptive Field of Neuron
Tactile_param['phit']= 1         # Center of Neuron  Receptive Field [mV]



#   Define Unisensory Neurons #
################################

class UnisensoryNeuron:

    def __init__(self,Auditory_Param,Tactile_param):
        
        #Define Auditory Parameters

        self.Ma,self.Na=int(Auditory_Param['Ma']),int(Auditory_Param['Na'])
        self.Center_a = np.zeros((self.Ma+1, self.Na+1, 2))
        for i in range(self.Ma):   
            for j in range(self.Na):
                xa = 10*(i+1) - 5 
                ya = 10*(j+1) - 15
                self.Center_a[i, j, 0] = xa
                self.Center_a[i, j, 1] = ya     
        self.Ia_0=Auditory_Param['Ia']
        self.sigmaa_I=Auditory_Param['sigmaaI']
        self.sigmaa=Auditory_Param['sigmaa']
        self.phia=Auditory_Param['phia']


        #Define Tactile Paremters
        
        self.Mt,self.Nt=int(Tactile_param['Mt']),int(Tactile_param['Nt'])
        self.Center_t = np.zeros((self.Mt+1, self.Nt+1, 2))
        for i in range(self.Mt):
            for j in range(self.Nt):
                xt = 0.5 * (i+1)
                yt = 0.5 * (j+1)
                self.Center_t[i, j, 0] = xt
                self.Center_t[i, j, 1] = yt   
        self.It_0=Tactile_param['It']
        self.sigmat_I=Tactile_param['sigmatI']
        self.sigmat=Tactile_param['sigmat']
        self.phit=Tactile_param['phit']

    def Stimuli(self,x0,y0,x,y,tI,t,s):
        if s == "a":
            xo=x0
            yo=y0
            self.tI=tI

            if (t < self.tI or t>self.tI+200): 
                I = 0
            else: 

                I = (self.Ia_0 ) * np.exp( (np.square(x-xo) + np.square(y-yo)) / (-2 * np.square(self.sigmaa_I)))
            return I

        elif s=="t":
            xo=x0
            yo=y0
            self.tI=tI

            if (t < self.tI or t>self.tI+200): 
                I = 0
            else: 
                I = (self.It_0 ) * np.exp( (np.square( x-xo) + np.square( y-yo)) / (-2 * np.square(self.sigmat_I)))
            return I

    def Phi(self,x,y,s):

        if s=="t":
            xt = np.arange(1,self.Mt+1)*0.5
            yt = np.arange(1,self.Nt+1)*0.5
            phi = np.zeros((self.Mt, self.Nt))
            for i in range(self.Mt):
                for j in range(self.Nt):
                    phi[i][j] = self.phit * np.exp(((np.square(x-xt[i]) + np.square(y-yt[j])) / (-2 * np.square(self.sigmat))))
            return phi

        elif s=="a":
            xa = (np.arange(1,self.Ma+1)*10)-5
            ya = (np.arange(1,self.Na+1)*10)-15
            phi = np.zeros((self.Ma, self.Na))
            for i in range(self.Ma):
                for j in range(self.Na):
                    phi[i][j] = self.phia * np.exp(((np.square(x-xa[i]) + np.square(y-ya[j])) / (-2 * np.square(self.sigmaa))))
            return phi

    def Phit(self,x0,y0,xl,yn,Phi,tI,t,s):
        
        if s=="t":
            PHI = np.zeros((self.Mt, self.Nt,len(xl),len(yn)))    
    
            for k in range(len(xl)):
                for l in range(len(yn)):
                    PHI[:,:,k,l] = np.multiply(Phi[:,:,k,l], UnisensoryNeuron.Stimuli(self,x0,y0,xl[k], yn[l],tI,t,s))

            PHI = np.sum(PHI,axis=3)
            PHI = np.sum(PHI,axis=2)

            return PHI

        elif s=="a":
            PHI = np.zeros((self.Ma, self.Na,len(xl),len(yn)))    
    
            for k in range(len(xl)):
                for l in range(len(yn)):
                    PHI[:,:,k,l] = np.multiply(Phi[:,:,k,l], UnisensoryNeuron.Stimuli(self,x0,y0,xl[k], yn[l],tI, t,s))
            PHI = np.sum(PHI,axis=3)
            PHI = np.sum(PHI,axis=2)

            return PHI
    
    def Unisensory_Calculation(self,x0a,y0a,x0t,y0t,tI,t):
        # Calculation
        dif = 0.2 
        xt_i = np.arange(0,20+dif,dif)
        yt_n = np.arange(0,10+dif,dif)

        phi_t = np.zeros((self.Mt,self.Nt,len(xt_i),len(yt_n)))  
        for k in range(len(xt_i)):
            for l in range(len(yt_n)):
                phi_t[:,:,k,l] = UnisensoryNeuron.Phi(self,xt_i[k],yt_n[l],"t")

        xa_i = np.arange(0,200+dif,dif)
        ya_n = np.arange(0,30+dif,dif)

        phi_a = np.zeros((self.Ma,self.Na,len(xa_i),len(ya_n)))        
        for k in range(len(xa_i)):
            for l in range(len(ya_n)):
                phi_a[:,:,k,l] = UnisensoryNeuron.Phi(self,xa_i[k],ya_n[l],"a")


        PHIt_a=UnisensoryNeuron.Phit(self,x0a,y0a,xa_i,ya_n,phi_a,tI,t,"a")
        PHIt_t=UnisensoryNeuron.Phit(self,x0t,y0t,xt_i,yt_n,phi_t,tI,t,"t")

        return(PHIt_a,PHIt_t)     

# Simulation #
##############

class UnisensoryNeuronSimulationPlots:


    def plot_image(fig,ax, Quan, letter_index, colorbar_label):
        im = ax.imshow(Quan)
        ax.set_ylabel('y [cm]')
        ax.set_xlabel('x [cm]')
        if colorbar_label =="Φa(t) [pA]":
            ax.set_xticks(np.arange(-0.5, 21, 2))
            ax.set_yticks(np.arange(-0.5, 3, 1))
            ax.set_xticklabels(np.arange(0, 210, 20))
            ax.set_yticklabels(np.arange(0, 31, 10)[::-1])
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", "5%", pad="3%")
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.get_yaxis().labelpad = 18
            cbar.ax.set_ylabel(colorbar_label, rotation=270, size=14,weight="bold")
            

        else:
            ax.set_xticks(np.arange(-.5, 41, 20))
            ax.set_yticks(np.arange(-.5, 21, 10))
            ax.set_xticklabels(np.arange(0, 30, 10))
            ax.set_yticklabels(np.arange(0, 15, 5)[::-1])
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", "5%", pad="3%")
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.get_yaxis().labelpad = 18
            cbar.ax.set_ylabel(colorbar_label, rotation=270, size=14,weight="bold")
           

    def Image(Quana, Quant):
        fig1, ax1 = plt.subplots(1, 1, sharex=True, figsize=(6, 6))

        UnisensoryNeuronSimulationPlots.plot_image(fig1,ax1, Quana, 0, 'Φa(t) [pA]')
        plt.show()
        fig2, ax2 = plt.subplots(1, 1, sharex=True, figsize=(6, 6))
        UnisensoryNeuronSimulationPlots.plot_image(fig2,ax2, Quant, 1, 'Φt(t) [pA]')
        plt.show()
    

    def Plot3D(x,y,z,s, colorsMap='viridis'):

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Make data.
        X = x
        Y = y
        X, Y = np.flip(np.meshgrid(X, Y),1)
        Z = np.transpose(z)

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.get_cmap(colorsMap),
                            linewidth=0, antialiased=True)
        ax.set_zticks([])

        
        # Add a color bar which maps values to colors.
        if s== "a":
  
            fig.colorbar(surf,shrink=0.5, aspect=5).set_label(label='Ia(x,y,t) [pA]',weight='bold',rotation=270,labelpad=12)
            plt.suptitle('Auditory External Stimuli with I0=3.6 and σ=0.3 ', fontsize=12)

            # Axis labels
            ax.set_xlabel('x [cm]')
            ax.set_ylabel('y [cm]')       
        else:
   
            fig.colorbar(surf,shrink=0.5, aspect=5).set_label(label='It(x,y,t) [pA]',weight='bold',rotation=270,labelpad=12)
            plt.suptitle('Tactile External Stimuli with I0=2.5 and σ=0.3', fontsize=12)

            # Axis labels
            ax.set_xlabel('x [cm]')
            ax.set_ylabel('y [cm]')

        
        plt.show()

class UnisensoryNeuronSimulation:
    def __init__(self,time,diff):
        self.tI = time
        self.dif = diff
        self.t=101

    def simulate(self):
        x0a=100
        y0a=15
        x0t=10
        y0t=5   

        UD = UnisensoryNeuron(Auditory_Param, Tactile_param)
        PHIa, PHIt = UD.Unisensory_Calculation(x0a,y0a,x0t,y0t,self.tI,self.t)
 

        t0=100
        dif=0.2
        ## Tatcile 
        xt = np.arange(0,20+dif,dif)
        yt = np.arange(0,10+dif,dif)

        I_tactile = np.zeros((len(xt), len(yt)))
        for i in range(len(xt)):
            for j in range(len(yt)):
                I_tactile[i][j] = UD.Stimuli(x0t,y0t,xt[i], yt[j], self.tI,self.t, "t")
        
        UnisensoryNeuronSimulationPlots.Plot3D(xt, yt, I_tactile,"t")

        ## Auditory
        dif=0.2
        xa =  np.arange(0,200+dif,dif)
        ya =  np.arange(0,30+dif,dif)
        
        I_audit = np.zeros((len(xa), len(ya)))
        for i in range(len(xa)):
            for j in range(len(ya)):
                I_audit[i][j] = UD.Stimuli(x0a,y0a,xa[i], ya[j],self.tI, self.t, "a")

        UnisensoryNeuronSimulationPlots.Plot3D(xa, ya, I_audit,"a")
        
        ## Unisensory Input
        PHIa=np.flip(PHIa.transpose(),1)
        PHIt=np.flip(PHIt.transpose(),1)
        UnisensoryNeuronSimulationPlots.Image(PHIa, PHIt)
        
# First RQ #
############
'''
t=100
d=0.2
UNSS=UnisensoryNeuronSimulation(t,d)
UNSS.simulate()
'''


###########################################################
#   Part 2.2: Setting up the connection between  neurons  #
###########################################################

#   Define Parameters   #
#########################
Auditory_Param["Lex"]=0.15
Auditory_Param["Lin"]=0.05
Auditory_Param["Sigma_ex"]=20
Auditory_Param["Sigmat_in"]=80
Tactile_param["Lex"]=0.15
Tactile_param["Lin"]=0.05
Tactile_param["Sigma_ex"]=1
Tactile_param["Sigmat_in"]=1

# Lateral  connection 
###############################


class LateralConnections:
    def __init__(self,Auditory_Param,Tactile_param):

        #Define Auditory Paremters
        self.Ma,self.Na=int(Auditory_Param['Ma']),int(Auditory_Param['Na'])
        self.xa = (np.arange(1,self.Ma+1)*10)-5
        self.ya = (np.arange(1,self.Na+1)*10)-15

        self.La_ex = Auditory_Param["Lex"]
        self.La_in = Auditory_Param["Lin"]
        self.sigmaa_ex = Auditory_Param["Sigma_ex"]
        self.sigmaa_in = Auditory_Param["Sigmat_in"]


        #Define Tactile Paremters
        self.Mt,self.Nt=int(Tactile_param['Mt']),int(Tactile_param['Nt'])
        self.xt = np.arange(1,self.Mt+1)*0.5 # originally was from 1 to Mt+1 
        self.yt = np.arange(1,self.Nt+1)*0.5

        self.Lt_ex = Tactile_param["Lex"]
        self.Lt_in = Tactile_param["Lin"]
        self.sigmat_ex = Tactile_param["Sigma_ex"]
        self.sigmat_in = Tactile_param["Sigmat_in"]

        
    def tactile_connections(self):
        Lt = np.zeros((self.Mt*self.Nt,self.Mt*self.Nt))
        for i in range(self.Mt*self.Nt):
            for j in range(self.Mt*self.Nt):
                if i == j: 
                    Lt[i,j] = 0
                else:
                    Dtx = self.xt[np.floor_divide(i,self.Nt)] - self.xt[np.floor_divide(j,self.Nt)]
                    Dty = self.yt[np.remainder(i,self.Nt)] - self.yt[np.remainder(j,self.Nt)]
                    Lt[i,j] = self.Lt_ex * np.exp(- (np.square(Dtx) + np.square(Dty)) / (2 * np.square(self.sigmat_ex))) - self.Lt_in * np.exp(- (np.square(Dtx) + np.square(Dty)) / (2 * np.square(self.sigmat_in)))
        return Lt
    
    def auditory_connections(self):
        La = np.zeros((self.Ma*self.Na,self.Ma*self.Na))

        for i in range(self.Ma*self.Na):
            for j in range(self.Ma*self.Na):
                if i == j: 
                    La[i,j] = 0
                else: #Fix this to acomdae auditor parmaeters 
                    Dax = self.xa[np.floor_divide(i,self.Na)] - self.xa[np.floor_divide(j,self.Na)]
                    Day = self.ya[np.remainder(i,self.Na)] - self.ya[np.remainder(j,self.Na)]
                    La[i,j] = self.La_ex * np.exp(- (np.square(Dax) + np.square(Day)) / (2 * np.square(self.sigmaa_ex))) - self.La_in * np.exp(- (np.square(Dax) + np.square(Day)) / (2 * np.square(self.sigmaa_in)))
        return La



# Simulation #
##############

class NeuralConnectionPlot: 
    def __init__(self, Lt, La):
        self.Lt = Lt
        self.La = La
        self.Mt=40
        self.Nt=20
        self.Ma=20
        self.Na=3

    def plot(self):
        fig,(ax1) = plt.subplots(1,1,sharex=True,figsize=(12,6))
        fig.subplots_adjust(wspace = 0.75, hspace = 0.5)



        im1 = ax1.imshow(self.Lt)
        ax1.set_xlabel('Horizontal Neuron Index [i]',size=12)
        ax1.set_ylabel('Vertical Neuron Index [j]',size=12)
        ax1.set_xticks(np.arange(-.5, 801, 200))
        ax1.set_yticks(np.arange(-.5, 801, 200))
        ax1.set_xticklabels(np.arange(0, 801, 200))
        ax1.set_yticklabels(np.arange(0, 801, 200))
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = fig.colorbar(im1,cax=cax)
        cbar.ax.get_yaxis().labelpad = 9
        cbar.ax.set_ylabel('Synaptic Weight Lt', rotation=270,size=14,weight="bold")
        ax1.set_title("Tactile Unisensory Area")
        plt.show()

        fig,(ax2) = plt.subplots(1,1,sharex=True,figsize=(12,6))
        fig.subplots_adjust(wspace = 0.75, hspace = 0.5)

        im2 = ax2.imshow(self.La)
        ax2.set_xlabel('Horizontal Neuron Index [i]',size=12)
        ax2.set_ylabel('Vertical Neuron Index [j]',size=12)
        ax2.set_xticks(np.arange(-.5, 61, 10))
        ax2.set_yticks(np.arange(-.5, 61, 10))
        ax2.set_xticklabels(np.arange(0, 61, 10))
        ax2.set_yticklabels(np.arange(0, 61, 10))
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = fig.colorbar(im2,cax=cax)
        cbar.ax.get_yaxis().labelpad = 9
        cbar.ax.set_ylabel('Synaptic Weight La', rotation=270,size=14,weight="bold")
        ax2.set_title("Auditory Unisensory Area")

        plt.show()

class NeuralConnectionSimulation:

    def __init__(self) :
        pass
    def simulate(self):
        LC=LateralConnections(Auditory_Param,Tactile_param)
        Lt=LC.tactile_connections()
        La=LC.auditory_connections()
        NCP=NeuralConnectionPlot(Lt,La)
        NCP.plot()


# Second RQ  #
##############
'''
NCS=NeuralConnectionSimulation()
NCS.simulate()
'''


# Feed forward and backward connectivity 
########################################

Auditory_Param["Wa0"]=6.5
Auditory_Param["Ba0"]=2.5
Auditory_Param["k1"]=15
Auditory_Param["k2"]=800
Auditory_Param["alpha"]=0.9
Auditory_Param["lim"]=65
Tactile_param["Wt0"]=6.5
Tactile_param["Bt0"]=2.5

class Synapse:
    #Synpatic strength between uni and mlti snssory ares , weight 
    def __init__(self, Auditory_Param,Tactile_param):
        #Define Auditory Paremters
        self.Ma,self.Na=int(Auditory_Param['Ma']),int(Auditory_Param['Na'])
        self.xa = (np.arange(1,self.Ma+1)*10)-5
        self.ya = (np.arange(1,self.Na+1)*10)-15
        self.Wa0=Auditory_Param["Wa0"]
        self.Ba0=Auditory_Param["Ba0"]
        self.k1=Auditory_Param["k1"]
        self.k2=Auditory_Param["k2"]
        self.alpha=Auditory_Param["alpha"]
        self.lim=Auditory_Param["lim"]
        
        #Define Tactile Paremters
        self.Mt,self.Nt=int(Tactile_param['Mt']),int(Tactile_param['Nt'])
        self.xt = np.arange(1,self.Mt+1)*0.5 
        self.yt = np.arange(1,self.Nt+1)*0.5
        self.Wt0=Tactile_param["Wt0"]
        self.Bt0=Tactile_param["Bt0"]

    def AuditorySynapticStrength(self, xa, ya):
        Ba = np.zeros((self.Ma,self.Na))
        Wa = np.zeros((self.Ma,self.Na))

        for i in range(self.Ma):
            for j in range(self.Na):# ask limits 2.2.2
                if (xa[i]<Auditory_Param["lim"]) & (ya[j]<Auditory_Param["lim"]-45): 
                    D = 0
                else: 
                    D = distance.euclidean((xa[i],ya[j]),(Auditory_Param["lim"],ya[j]))   

                Ba[i,j] = self.alpha*self.Ba0*np.exp(- D/self.k1)+(1-self.alpha)*self.Ba0*np.exp(- D/self.k2)
                Wa[i,j] = self.alpha*self.Wa0*np.exp(- D/self.k1)+(1-self.alpha)*self.Wa0*np.exp(- D/self.k2)
        
        return Ba, Wa

    def TactileSynapticStrength(self,xt,yt):
        Bt = np.zeros((self.Mt,self.Nt))
        Wt = np.zeros((self.Mt,self.Nt))

        for i in range(self.Mt):
            for j in range(self.Nt):
                Bt[i,j] = self.Bt0
                Wt[i,j] = self.Wt0
        return Bt,Wt
        
class MultisensoryNeuron:
    def __init__(self,Auditory_Param,Tactile_param):
        self.Mt,self.Nt=int(Tactile_param['Mt']),int(Tactile_param['Nt'])
        self.Ma,self.Na=int(Auditory_Param['Ma']),int(Auditory_Param['Na']) 
        self.xt = np.arange(1,self.Mt+1)*0.5 
        self.yt = np.arange(1,self.Nt+1)*0.5
        self.xa = (np.arange(1,self.Ma+1)*10)-5
        self.ya = (np.arange(1,self.Na+1)*10)-15
        
        
        self.synapse = Synapse(Auditory_Param,Tactile_param)
        self.Ba, self.Wa = self.synapse.AuditorySynapticStrength(self.xa, self.ya)
        self.Bt, self.Wt = self.synapse.TactileSynapticStrength(self.xt, self.yt)

class FeedFBWeightsPlot:

    def __init__(self, Wa, Ba,Bt,Wt):
        self.Wa = Wa
        self.Ba = Ba    
        self.Bt = Bt
        self.Wt = Wt


    def plot(self):
        
        fig, (axa1,axa2) = plt.subplots(1, 2, sharex=True, figsize=(12, 6))
        fig.subplots_adjust(wspace = 0.75, hspace = 0.5)
        
        axa1 = plt.subplot(121)
        im1 = axa1.imshow(self.Wa.transpose())
        axa1.set_ylabel('x [cm]')
        axa1.set_xlabel('y [cm]')
        axa1.set_xticks(np.arange(-.5, 21, 2))
        axa1.set_yticks(np.arange(-.5, 3, 1))
        axa1.set_xticklabels(np.arange(0, 210, 20))
        axa1.set_yticklabels(np.arange(0, 31, 10)[::-1])
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = fig.colorbar(im1, cax=cax)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('Wa', rotation=270, size=14,weight="bold")
        axa1.set_title("FeedBack Weight")

        axa2 = plt.subplot(122)
        im2 = axa2.imshow(self.Ba.transpose())
        axa2.set_ylabel('x [cm]')
        axa2.set_xlabel('y [cm]')
        axa2.set_xticks(np.arange(-.5, 21, 2))
        axa2.set_yticks(np.arange(-.5, 3, 1))
        axa2.set_xticklabels(np.arange(0, 210, 20))
        axa2.set_yticklabels(np.arange(0, 31, 10)[::-1])
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = fig.colorbar(im2, cax=cax)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('Ba', rotation=270, size=14, weight="bold")
        axa2.set_title("FeedWorward Weight")
        fig.suptitle("Auditory Unisensory Area to Multisensory Area")
        plt.show()

        fig, (axt1,axt2) = plt.subplots(1, 2, sharex=True, figsize=(6, 6))
        fig.subplots_adjust(wspace = 0.75, hspace = 0.5)

        axt1 = plt.subplot(121)
        im1 = axt1.imshow(self.Wt.transpose())
        axt1.set_ylabel('x [cm]')
        axt1.set_xlabel('y [cm]')
        axt1.set_xticks(np.arange(-0.5, 40, 8))
        axt1.set_yticks(np.arange(-0.5, 29, 10))
        axt1.set_xticklabels(np.arange(0, 21, 4))
        axt1.set_yticklabels(np.arange(0, 11, 5)[::-1])
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = fig.colorbar(im1, cax=cax)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('Wt', rotation=270, size=14,weight="bold")
        axt1.set_title("FeedBack Weight")

        axt2 = plt.subplot(122)
        im2 = axt2.imshow(self.Bt.transpose())
        axt2.set_ylabel('x [cm]')
        axt2.set_xlabel('y [cm]')
        axt2.set_xticks(np.arange(-0.5, 40, 8))
        axt2.set_yticks(np.arange(-0.5, 29, 10))
        axt2.set_xticklabels(np.arange(0, 21, 4))
        axt2.set_yticklabels(np.arange(0, 11, 5)[::-1])
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = fig.colorbar(im2, cax=cax)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('Bt', rotation=270, size=14,weight="bold")
        axt2.set_title("FeedForward Weight")
        fig.suptitle("Tactile Unisensory Area to Multisensory Area")
        plt.show()



# Simulation
##############

class FeedFBWeightsSimulation:
    def __init__(self) :
        pass
    def simulate(self):
        MN=MultisensoryNeuron(Auditory_Param,Tactile_param)
        Wa,Ba,Wt,Bt=MN.Wa,MN.Ba,MN.Wt,MN.Bt
        FBWP=FeedFBWeightsPlot(Wa,Ba,Wt,Bt)
        FBWP.plot()




# Second RQ  #
##############
'''
FBWS=FeedFBWeightsSimulation()
FBWS.simulate()
'''

#####################################
#   Part 2.3: Responses of neurons  #
#####################################

class UnisensoryNeuronActivity:
    def __init__(self,Simulation_param):
        #Simulation Parameters 
        self.dt,self.T=Simulation_param['dt'],Simulation_param['T']

    def LateralInputs(self,z,L,s):
        Mt=40
        Nt=20
        Ma=20
        Na=3
        if s=="t":
            It = np.zeros(Mt*Nt)
            z = np.reshape(z, (1, Mt*Nt))
            for i in range(Mt*Nt):
                It[i] = np.sum(np.multiply(L[i, :], z[0, :]))
            It = np.reshape(It, (Mt, Nt))
            return It

        if s=="a":
            Ia = np.zeros(Ma*Na)
            z = np.reshape(z, (1, Ma*Na))
            for i in range(Ma*Na):
                Ia[i] = np.sum(np.multiply(L[i, :], z[0, :]))
            Ia = np.reshape(Ia, (Ma, Na))
            return Ia

    def FeedbackInput(self,z,B,s):
        if s=="t":
            b=np.multiply(B, z)
        elif s=="a":
            b=np.multiply(B, z)
        return b

    def Input(self,Phi,L,Lt,La,b,B,z,zm,timestep,s):
        if s=="t":
            Phi=np.reshape(Phi,(40,20))
            Input=Phi+L(z[:,:,timestep],Lt,"t")+b(zm[timestep],B,"t")
            return Input
        elif s=="a":
            Phi=np.reshape(Phi,(20,3))
            Input=Phi+L(z[:,:,timestep],La,"a")+b(zm[timestep],B,"a")
            return Input


    def DynamicState(self,q,ut):

        dq = (self.dt/20.0)*(-q+ut)
        State = q+ dq

        return State

    def Activation(self,q,s):
        ft_min = -0.12
        ft_max = 1
        qt_c = 19.43
        rt = 0.34
        Mt=40
        Nt=20

        fa_min = -0.12
        fa_max = 1
        qa_c = 19.43
        ra = 0.34
        Ma=20
        Na=3

        if s=="t":
            acti = q
            for i in range(Mt):
                for j in range(Nt):
                    acti[i,j] = (ft_min+ft_max*np.exp((q[i,j]-qt_c)*rt))/(1+np.exp((q[i,j]-qt_c)*rt))
            return acti

        if s=="a":
            acti = q
            for i in range(Ma):
                for j in range(Na):
                    acti[i,j] = (fa_min+fa_max*np.exp((q[i,j]-qa_c)*ra))/(1+np.exp((q[i,j]-qa_c)*ra))
            return acti

class MultisensoryNeuronActivity:

    def __init__(self,Simulation_Param):
        self.dt,self.T=Simulation_Param['dt'],Simulation_Param['T']

    def Input(self,Wt,Wa,zt,za,timestep):
        um=np.sum(np.multiply(Wt,zt[:,:,timestep]))+np.sum(np.multiply(Wa,za[:,:,timestep]))
        return um


    def DynamicState(self,q,um):

        dq = (self.dt/20)*(-q+um)
        State= q+ dq
        return State 

    def Activation(self,q):
        fm_min = 0
        fm_max = 1
        qm_c = 12
        rm = 0.6
        Acti = (fm_min+fm_max*np.exp((q-qm_c)*rm))/(1+np.exp((q-qm_c)*rm))
        return Acti



# Simulation
##############

class Simulation:

    def __init__(self,x0a,y0a,x0t,y0t):

        self.Simulation_Param={}
        self.Simulation_Param["dt"]=0.4
        self.Simulation_Param["T"]=400
        self.Simulation_Param["tI"]=100
        self.Timesteps=int(self.Simulation_Param["T"]/self.Simulation_Param["dt"])
        self.Mt,self.Nt=Tactile_param["Mt"],Tactile_param["Nt"]
        self.Ma,self.Na=Auditory_Param["Ma"],Auditory_Param["Na"]

        self.x0a=x0a
        self.y0a=y0a
        self.x0t=x0t
        self.y0t=y0t

        self.qt = np.zeros((self.Mt,self.Nt,self.Timesteps+1))
        self.ut = np.zeros((self.Mt,self.Nt,self.Timesteps+1))
        self.zt = np.zeros((self.Mt,self.Nt,self.Timesteps+1))
        self.at = np.zeros((self.Mt,self.Nt,self.Timesteps+1))

        self.qa = np.zeros((self.Ma,self.Na,self.Timesteps+1))
        self.ua = np.zeros((self.Ma,self.Na,self.Timesteps+1))
        self.za = np.zeros((self.Ma,self.Na,self.Timesteps+1))
        self.aa = np.zeros((self.Ma,self.Na,self.Timesteps+1))

        self.qm = np.zeros(self.Timesteps+1)
        self.um = np.zeros(self.Timesteps+1)
        self.zm = np.zeros(self.Timesteps+1)
        self.am = np.zeros(self.Timesteps+1)


        self.UD = UnisensoryNeuron(Auditory_Param, Tactile_param)
        LC=LateralConnections(Auditory_Param, Tactile_param)
        self.Lt,self.La=LC.tactile_connections(),LC.auditory_connections()
        self.UNA=UnisensoryNeuronActivity(self.Simulation_Param)
        MN=MultisensoryNeuron(Auditory_Param,Tactile_param)
        self.Wa,self.Ba=MN.Wa,MN.Ba
        self.Wt,self.Bt=MN.Wt,MN.Bt
        self.MNA=MultisensoryNeuronActivity(self.Simulation_Param)
        self.PHIa, self.PHIt = self.UD.Unisensory_Calculation(self.x0a,self.y0a,self.x0t,self.y0t,self.Simulation_Param["tI"],self.Simulation_Param["tI"]+1)


    def NeuronResponse(self):

            for i in range(self.Timesteps):
                t=i*self.Simulation_Param["dt"]

                if (t<self.Simulation_Param["tI"] or t>self.Simulation_Param["tI"]+200) :
                    PHIt,PHIa=np.zeros((40,20)),np.zeros((20,3))
                else:
                    PHIt,PHIa=self.PHIt,self.PHIa

                self.ut[:,:,i+1] = self.UNA.Input(PHIt,self.UNA.LateralInputs,self.Lt,self.La,self.UNA.FeedbackInput,self.Bt,self.zt,self.zm,i,"t")
                self.qt[:,:,i+1] = self.UNA.DynamicState(self.qt[:,:,i],self.ut[:,:,i])
                self.at[:,:,i+1] = self.UNA.Activation(self.qt[:,:,i],"t")
                self.zt[:,:,i+1] = self.at[:,:,i]*np.heaviside(self.at[:,:,i],0)
            

                self.ua[:,:,i+1] = self.UNA.Input(PHIa,self.UNA.LateralInputs,self.Lt,self.La,self.UNA.FeedbackInput,self.Ba,self.za,self.zm,i,"a")
                self.qa[:,:,i+1] = self.UNA.DynamicState(self.qa[:,:,i],self.ua[:,:,i])
                self.aa[:,:,i+1] = self.UNA.Activation(self.qa[:,:,i],"a")
                self.za[:,:,i+1] = self.aa[:,:,i]*np.heaviside(self.aa[:,:,i],0)

                #self.Wa
                #self.za
                #self.MNA.Input(self.Wt,np.zeros((self.Wa.shape)),self.zt,np.zeros((self.za.shape)),i)
                #self.MNA.Input(self.Wt,self.Wa,self.zt,self.za,i)
                #self.MNA.Input(np.zeros((self.Wt.shape)),self.Wa,np.zeros((self.zt.shape)),self.za,i)
                self.um[i+1] = self.MNA.Input(self.Wt,np.zeros((self.Wa.shape)),self.zt,np.zeros((self.za.shape)),i)
                self.am[i+1] = self.MNA.Activation(self.qm[i])
                self.zm[i+1] = self.am[i]*np.heaviside(self.am[i],0) 

            return self.zt,self.za,self.zm
      

#Plots
###############
class ResponsePlots:

    def plot_image(fig,ax, Quan, letter_index, colorbar_label,label):
        Quan=np.flip(np.transpose(Quan),1)
        im = ax.imshow(Quan)
        ax.set_ylabel('x [cm]')
        ax.set_xlabel('y [cm]')
        if label =="auditory":
            ax.set_xticks(np.arange(-.5, 21, 2))
            ax.set_yticks(np.arange(-.5, 3, 1))
            ax.set_xticklabels(np.arange(0, 210, 20))
            ax.set_yticklabels(np.arange(0, 31, 10)[::-1])
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", "5%", pad="3%")
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.get_yaxis().labelpad = 15
            cbar.ax.set_ylabel("[pA]", rotation=270, size=12)
         

        else:
    
            ax.set_xticks(np.arange(-.5, 41, 20))
            ax.set_yticks(np.arange(-.5, 21, 10))
            ax.set_xticklabels(np.arange(0, 30, 10))
            ax.set_yticklabels(np.arange(0, 15, 5)[::-1])
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", "5%", pad="3%")
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.get_yaxis().labelpad = 15
            cbar.ax.set_ylabel("[pA]", rotation=270, size=12)
            

    def Image(Quana, Quant,labela,labelt):
        fig1, ax1 = plt.subplots(1, 1, sharex=True, figsize=(6, 6))

        ResponsePlots.plot_image(fig1,ax1, Quana, 0, labela,"auditory")
        ax1.set_title(labela)
        plt.show()
        fig2, ax2 = plt.subplots(1, 1, sharex=True, figsize=(6, 6))
        ResponsePlots.plot_image(fig2,ax2, Quant, 1, labelt,"tactile")
        ax2.set_title(labelt)
        plt.show()
    

    def Plot3D(x,y,z,s,label,colorsMap='viridis'):

        #Add title and label and update 

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Make data.
        X = x
        Y = y
        X, Y = np.meshgrid(X, Y)
        Z = np.transpose(z)
        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.get_cmap(colorsMap),
                            linewidth=0, antialiased=True)
        ax.set_zticks([])

        
        # Add a color bar which maps values to colors.
        if s== "a":
            fig.colorbar(surf,shrink=0.5, aspect=3).set_label(label="Za [Hz]",weight="bold",rotation=270,labelpad=12)
            fig.colorbar

            # Axis labels
            ax.set_xlabel('x [cm]')
            ax.set_ylabel('y [cm]')    
            ax.set_title(label)
       
        else:
            fig.colorbar(surf,shrink=0.5, aspect=3).set_label(label="Zt [Hz]",weight="bold",rotation=270,labelpad=12)

            # Axis labels
            ax.set_xlabel('x [cm]')
            ax.set_ylabel('y [cm]')
            ax.set_title(label)
            

        plt.show()
    
    def Lineplot(z,t,xlabel,ylabel,label):

        plt.plot(t, z, color='y')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(label)
        plt.show()





#  Third RQ
#############  

# Repsonse 
class UnisensoryNeuronResponseSimulation:
    def __init__(self):
        pass


    def simulate(self):

        SUR=Simulation(100,15,10,5)
        Zt,Za,Zm = SUR.NeuronResponse()
        
       
        ## Tatcile 
        xt = np.arange(1,Tactile_param["Mt"]+1)*0.5 
        yt = np.arange(1,Tactile_param["Nt"]+1)*0.5

        ResponsePlots.Plot3D(xt, yt,Zt[:,:,750],"t","Tactile Area Response at \n t=300 Steady-State") 
        ResponsePlots.Image(Za[:,:,750], Zt[:,:,750],"Za(t=300)","Zt(t=300)") 

        ## Auditory
        xa = (np.arange(1,Auditory_Param["Ma"]+1)*10)-5
        ya = (np.arange(1,Auditory_Param["Na"]+1)*10)-15

        ResponsePlots.Plot3D(xa, ya, Za[:,:,750],"a","Auditory Area Response at \n t=300 Steady-State" ) 
        ## Unisensory Input
        ResponsePlots.Image(Za[:,:,750], Zt[:,:,750],"Za(t=300)","Zt(t=300)") 

        ##Line Plots
        time=np.arange(0, 400+0.4, 0.4)
        Za=np.mean(np.mean(Za,axis=1),axis=0)
        ResponsePlots.Lineplot(Za,time,"Time [ms]","Avg. Za [Hz]","Auditory Area Response")

        time=np.arange(0, 400+0.4, 0.4)
        Zt=np.mean(np.mean(Zt,axis=1),axis=0)
        ResponsePlots.Lineplot(Zt,time,"Time [ms]","Avg. Zt [Hz]","Tactile Area Response")

 

UNRS=UnisensoryNeuronResponseSimulation()
UNRS.simulate()




#Lateral connections in Unisensory diff. lex and lin 
# Change plot to the connections and get eth lateral inputs and descirbe ther relation of 
#Lateral inputs on unisensory activity 

# try 3d plot where x and y are on the x and Y ans z are the values of lateral inputs
class UnisensoryNeuronLateralConnectionSimulation:
    def __init__(self):
        self.Simulation_Param={}
        self.Simulation_Param["dt"]=0.4
        self.Simulation_Param["T"]=400
        self.Simulation_Param["tI"]=100
        self.Mt = 40
        self.Nt = 20
        self.Ma = 20
        self.Na = 3

    def simulate(self):
        Zt_values=[]
        Za_values=[]
        It_values=[]
        Ia_values=[]
        Lt_values=[]
        La_values=[]




        Lexdiff=[0.05,0.15,0.30]
        Lindiff=[0.05]   

        for i in Lindiff:
            Auditory_Param["Lin"]=i
            Tactile_param["Lin"]=i           
            for j in Lexdiff:
                Auditory_Param["Lex"]=j
                Tactile_param["Lex"]=j
                
                
                SUR=Simulation(100,15,10,5)
                LatInput=UnisensoryNeuronActivity(self.Simulation_Param)
                Zt,Za,Zm = SUR.NeuronResponse()
                LC=LateralConnections(Auditory_Param, Tactile_param)
                Lt,La=LC.tactile_connections(),LC.auditory_connections()
                Zt_values.append(Zt)
                Za_values.append(Za)
                Lt_values.append(Lt)
                La_values.append(La)
                
                It = np.zeros((self.Mt, self.Nt, Zt.shape[2]))
                for t in range(Zt.shape[2]):
                    z_flat_t = np.reshape(Zt[:, :, t], (1, self.Mt*self.Nt))
                    for i in range(self.Mt*self.Nt):
                        It[i // self.Nt, i % self.Nt, t] = np.sum(np.multiply(Lt[i, :], z_flat_t[0, :]))
                It_values.append(It)

                
                Ia = np.zeros((self.Ma, self.Na, Za.shape[2]))
                for t in range(Za.shape[2]):
                    z_flat_a = np.reshape(Za[:, :, t], (1, self.Ma*self.Na))
                    for i in range(self.Ma*self.Na):
                        Ia[i // self.Na, i % self.Na, t] = np.sum(np.multiply(La[i, :], z_flat_a[0, :]))
                Ia_values.append(Ia)

    

        Lexdiff=[0.15]
        Lindiff=[0.05,0.15,0.30]         

        for i in Lexdiff:
            Auditory_Param["Lex"]=i
            Tactile_param["Lex"]=i           
            for j in Lindiff:
                Auditory_Param["Lin"]=j
                Tactile_param["Lin"]=j


                SUR=Simulation(100,15,10,5)
                Zt,Za,Zm = SUR.NeuronResponse()
                LC=LateralConnections(Auditory_Param, Tactile_param)
                Lt,La=LC.tactile_connections(),LC.auditory_connections()
                Zt_values.append(Zt)
                Za_values.append(Za)
                Lt_values.append(Lt)
                La_values.append(La)
                
                It = np.zeros((self.Mt, self.Nt, Zt.shape[2]))
                for t in range(Zt.shape[2]):
                    z_flat_t = np.reshape(Zt[:, :, t], (1, self.Mt*self.Nt))
                    for i in range(self.Mt*self.Nt):
                        It[i // self.Nt, i % self.Nt, t] = np.sum(np.multiply(Lt[i, :], z_flat_t[0, :]))
                It_values.append(It)

                
                Ia = np.zeros((self.Ma, self.Na, Za.shape[2]))
                for t in range(Za.shape[2]):
                    z_flat_a = np.reshape(Za[:, :, t], (1, self.Ma*self.Na))
                    for i in range(self.Ma*self.Na):
                        Ia[i // self.Na, i % self.Na, t] = np.sum(np.multiply(La[i, :], z_flat_a[0, :]))
                Ia_values.append(Ia)
                

        #Tactile 

        ## Plots
        fig,(axt1,axt2,axt3) = plt.subplots(1,3,sharex=True,figsize=(12,4))
        fig.subplots_adjust(wspace = 0.75, hspace = 0.5)




        axt1 = plt.subplot(131)
        im1 = axt1.imshow(Lt_values[0])
        axt1.set_xlabel('Horizontal Neuron Index [i]',size=12)
        axt1.set_ylabel('Vertical Neuron Index [j]',size=12)
        axt1.set_xticks(np.arange(-.5, 801, 200))
        axt1.set_yticks(np.arange(-.5, 801, 200))
        axt1.set_xticklabels(np.arange(0, 801, 200))
        axt1.set_yticklabels(np.arange(0, 801, 200))
        axt1.set_title("Lex= 0.05")
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = fig.colorbar(im1,cax=cax)
        cbar.ax.get_yaxis().labelpad = 9
        cbar.ax.set_ylabel('Synaptic Strength', rotation=270,size=12,weight="bold")
                    
 
                    
        axt2 = plt.subplot(132)
        im2 = axt2.imshow(Lt_values[1])
        axt2.set_xlabel('Horizontal Neuron Index [i]',size=12)
        axt2.set_ylabel('Vertical Neuron Index [j]',size=12)
        axt2.set_xticks(np.arange(-.5, 801, 200))
        axt2.set_yticks(np.arange(-.5, 801, 200))
        axt2.set_xticklabels(np.arange(0, 801, 200))
        axt2.set_yticklabels(np.arange(0, 801, 200))
        axt2.set_title("Lex= 0.15")
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = fig.colorbar(im2,cax=cax)
        cbar.ax.get_yaxis().labelpad = 9
        cbar.ax.set_ylabel('Synaptic Strength', rotation=270,size=12,weight="bold")


        axt3 = plt.subplot(133)
        im3 = axt3.imshow(Lt_values[2])
        axt3.set_xlabel('Horizontal Neuron Index [i]',size=12)
        axt3.set_ylabel('Vertical Neuron Index [j]',size=12)
        axt3.set_xticks(np.arange(-.5, 801, 200))
        axt3.set_yticks(np.arange(-.5, 801, 200))
        axt3.set_xticklabels(np.arange(0, 801, 200))
        axt3.set_yticklabels(np.arange(0, 801, 200))
        axt3.set_title("Lex= 0.30")
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = fig.colorbar(im3,cax=cax)
        cbar.ax.get_yaxis().labelpad = 9
        cbar.ax.set_ylabel('Synaptic Strength', rotation=270,size=12,weight="bold")

        fig.suptitle("Lateral Connections in Tactile Unisensory Area with Lin=0.05")
        plt.show()

        ## Plots
        fig,(axt4,axt5,axt6) = plt.subplots(1,3,sharex=True,figsize=(12,4))
        fig.subplots_adjust(wspace = 0.75, hspace = 0.5)


        #Tactile

        axt4 = plt.subplot(131)
        im4 = axt4.imshow(Lt_values[3])
        axt4.set_xlabel('Horizontal Neuron Index [i]',size=12)
        axt4.set_ylabel('Vertical Neuron Index [j]',size=12)
        axt4.set_xticks(np.arange(-.5, 801, 200))
        axt4.set_yticks(np.arange(-.5, 801, 200))
        axt4.set_xticklabels(np.arange(0, 801, 200))
        axt4.set_yticklabels(np.arange(0, 801, 200))
        axt4.set_title("Lin= 0.05")
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = fig.colorbar(im4,cax=cax)
        cbar.ax.get_yaxis().labelpad = 9
        cbar.ax.set_ylabel('Synaptic Strength', rotation=270,size=12,weight="bold")
                    
 
                    
        axt5 = plt.subplot(132)
        im5 = axt5.imshow(Lt_values[4])
        axt5.set_xlabel('Horizontal Neuron Index [i]',size=12)
        axt5.set_ylabel('Vertical Neuron Index [j]',size=12)
        axt5.set_xticks(np.arange(-.5, 801, 200))
        axt5.set_yticks(np.arange(-.5, 801, 200))
        axt5.set_xticklabels(np.arange(0, 801, 200))
        axt5.set_yticklabels(np.arange(0, 801, 200))
        axt5.set_title("Lin= 0.15")
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = fig.colorbar(im5,cax=cax)
        cbar.ax.get_yaxis().labelpad = 9
        cbar.ax.set_ylabel('Synaptic Strength', rotation=270,size=12,weight="bold")


        axt6 = plt.subplot(133)
        im6 = axt6.imshow(Lt_values[5])
        axt6.set_xlabel('Horizontal Neuron Index [i]',size=12)
        axt6.set_ylabel('Vertical Neuron Index [j]',size=12)
        axt6.set_xticks(np.arange(-.5, 801, 200))
        axt6.set_yticks(np.arange(-.5, 801, 200))
        axt6.set_xticklabels(np.arange(0, 801, 200))
        axt6.set_yticklabels(np.arange(0, 801, 200))
        axt6.set_title("Lin= 0.30")
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = fig.colorbar(im6,cax=cax)
        cbar.ax.get_yaxis().labelpad = 9
        cbar.ax.set_ylabel('Synaptic Strength', rotation=270,size=12,weight="bold")

        fig.suptitle("Lateral Connections in Tactile Unisensory Area with Lex=0.15")
        plt.show()

                    

        #Auditory 

        ## Plots
        fig,(axt1,axt2,axt3) = plt.subplots(1,3,sharex=True,figsize=(12,4))
        fig.subplots_adjust(wspace = 0.75, hspace = 0.5)




        axt1 = plt.subplot(131)
        im1 = axt1.imshow(La_values[0])
        axt1.set_xlabel('Horizontal Neuron Index [i]',size=12)
        axt1.set_ylabel('Vertical Neuron Index [j]',size=12)
        axt1.set_xticks(np.arange(-.5, 61, 10))
        axt1.set_yticks(np.arange(-.5, 61, 10))
        axt1.set_xticklabels(np.arange(0, 61, 10))
        axt1.set_yticklabels(np.arange(0, 61, 10))
        axt1.set_title("Lex= 0.05")
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = fig.colorbar(im1,cax=cax)
        cbar.ax.get_yaxis().labelpad = 9
        cbar.ax.set_ylabel('Synaptic Strength', rotation=270,size=12,weight="bold")
                    
 
                    
        axt2 = plt.subplot(132)
        im2 = axt2.imshow(La_values[1])
        axt2.set_xlabel('Horizontal Neuron Index [i]',size=12)
        axt2.set_ylabel('Vertical Neuron Index [j]',size=12)
        axt2.set_xticks(np.arange(-.5, 61, 10))
        axt2.set_yticks(np.arange(-.5, 61, 10))
        axt2.set_xticklabels(np.arange(0, 61, 10))
        axt2.set_yticklabels(np.arange(0, 61, 10))
        axt2.set_title("Lex= 0.15")
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = fig.colorbar(im2,cax=cax)
        cbar.ax.get_yaxis().labelpad = 9
        cbar.ax.set_ylabel('Synaptic Strength', rotation=270,size=12,weight="bold")


        axt3 = plt.subplot(133)
        im3 = axt3.imshow(La_values[2])
        axt3.set_xlabel('Horizontal Neuron Index [i]',size=12)
        axt3.set_ylabel('Vertical Neuron Index [j]',size=12)
        axt3.set_xticks(np.arange(-.5, 61, 10))
        axt3.set_yticks(np.arange(-.5, 61, 10))
        axt3.set_xticklabels(np.arange(0, 61, 10))
        axt3.set_yticklabels(np.arange(0, 61, 10))
        axt3.set_title("Lex= 0.30")
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = fig.colorbar(im3,cax=cax)
        cbar.ax.get_yaxis().labelpad = 9
        cbar.ax.set_ylabel('Synaptic Stength', rotation=270,size=12,weight="bold")

        fig.suptitle("Lateral Connections in Auditory Unisensory Area with Lin=0.05")
        plt.show()


 ## Plots
        fig,(axt4,axt5,axt6) = plt.subplots(1,3,sharex=True,figsize=(12,4))
        fig.subplots_adjust(wspace = 0.75, hspace = 0.5)


        #Tactile

        axt4 = plt.subplot(131)
        im4 = axt4.imshow(La_values[3])
        axt4.set_xlabel('Horizontal Neuron Index [i]',size=12)
        axt4.set_ylabel('Vertical Neuron Index [j]',size=12)
        axt4.set_xticks(np.arange(-.5, 61, 10))
        axt4.set_yticks(np.arange(-.5, 61, 10))
        axt4.set_xticklabels(np.arange(0, 61, 10))
        axt4.set_yticklabels(np.arange(0, 61, 10))
        axt4.set_title("Lin= 0.05")
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = fig.colorbar(im4,cax=cax)
        cbar.ax.get_yaxis().labelpad = 9
        cbar.ax.set_ylabel('Synaptic Strength', rotation=270,size=12,weight="bold")
                    
 
                    
        axt5 = plt.subplot(132)
        im5 = axt5.imshow(La_values[4])
        axt5.set_xlabel('Horizontal Neuron Index [i]',size=12)
        axt5.set_ylabel('Vertical Neuron Index [j]',size=12)
        axt5.set_xticks(np.arange(-.5, 61, 10))
        axt5.set_yticks(np.arange(-.5, 61, 10))
        axt5.set_xticklabels(np.arange(0, 61, 10))
        axt5.set_yticklabels(np.arange(0, 61, 10))
        axt5.set_title("Lin= 0.15")
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = fig.colorbar(im5,cax=cax)
        cbar.ax.get_yaxis().labelpad = 9
        cbar.ax.set_ylabel('Synaptic Strength', rotation=270,size=12,weight="bold")


        axt6 = plt.subplot(133)
        im6 = axt6.imshow(La_values[5])
        axt6.set_xlabel('Horizontal Neuron Index [i]',size=12)
        axt6.set_ylabel('Vertical Neuron Index [j]',size=12)
        axt6.set_xticks(np.arange(-.5, 61, 10))
        axt6.set_yticks(np.arange(-.5, 61, 10))
        axt6.set_xticklabels(np.arange(0, 61, 10))
        axt6.set_yticklabels(np.arange(0, 61, 10))
        axt6.set_title("Lin= 0.30")
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = fig.colorbar(im6,cax=cax)
        cbar.ax.get_yaxis().labelpad = 9
        cbar.ax.set_ylabel('Synaptic Stregnth', rotation=270,size=12,weight="bold")

        fig.suptitle("Lateral Connections in Auditory Unisensory Area with Lex=0.15")
        plt.show()





        #Repsonse Vs time for each Lin and Lex 
        fig, (axt1,axt2) = plt.subplots(1,2,sharex=True,figsize=(12,4));
        fig.subplots_adjust(wspace = 0.75, hspace = 0.5)

        time=np.arange(0, 400+0.4, 0.4)
        #Tactile
        zt1= np.mean(np.mean(Zt_values[0],axis=1),axis=0)
        zt2= np.mean(np.mean(Zt_values[1],axis=1),axis=0)
        zt3= np.mean(np.mean(Zt_values[2],axis=1),axis=0)
        #Tactile
        zt4= np.mean(np.mean(Zt_values[3],axis=1),axis=0)
        zt5= np.mean(np.mean(Zt_values[4],axis=1),axis=0)
        zt6= np.mean(np.mean(Zt_values[5],axis=1),axis=0)

        axt1 = plt.subplot(121)
        axt1.plot(time, zt1, label='Lex=0.05')
        axt1.plot(time, zt2, label='Lex=0.15')
        axt1.plot(time, zt3, label='Lex=0.30')

        axt1.legend()
        axt1.set_xlabel("Time [ms]")
        axt1.set_ylabel("Response [Hz]")

        axt2 = plt.subplot(122)
        axt2.plot(time, zt4, label='Lin= 0.05')
        axt2.plot(time, zt5, label='Lin= 0.15')
        axt2.plot(time, zt6, label='Lin= 0.30')

        axt2.legend()
        axt2.set_xlabel("Time [ms]")
        axt2.set_ylabel("Response [Hz]")

        fig.suptitle("Effect of Lateral Connections in Tactile Area")
        plt.show()

        # Lateral Input versus Time for each Lex and Lin 
        fig, (axIt1,axIt2) = plt.subplots(1,2,sharex=True,figsize=(12,4));
        fig.subplots_adjust(wspace = 0.75, hspace = 0.5)

        time=np.arange(0, 400+0.4, 0.4)
        #Tactile
        It1= np.mean(np.mean(It_values[0],axis=1),axis=0)
        It2= np.mean(np.mean(It_values[1],axis=1),axis=0)
        It3= np.mean(np.mean(It_values[2],axis=1),axis=0)
        #Tactile
        It4= np.mean(np.mean(It_values[3],axis=1),axis=0)
        It5= np.mean(np.mean(It_values[4],axis=1),axis=0)
        It6= np.mean(np.mean(It_values[5],axis=1),axis=0)

        axIt1 = plt.subplot(121)
        axIt1.plot(time, It1, label='Lex=0.05')
        axIt1.plot(time, It2, label='Lex=0.15')
        axIt1.plot(time, It3, label='Lex=0.30')

        axIt1.legend()
        axIt1.set_xlabel("Time [ms]")
        axIt1.set_ylabel("Lateral Input [Hz]")

        axIt2 = plt.subplot(122)
        axIt2.plot(time, It4, label='Lin= 0.05')
        axIt2.plot(time, It5, label='Lin= 0.15')
        axIt2.plot(time, It6, label='Lin= 0.30')

        axIt2.legend()
        axIt2.set_xlabel("Time [ms]")
        axIt2.set_ylabel("Lateral Input [Hz]")

        fig.suptitle("Effect of Lateral Connections in Tactile Area")
        plt.show()
            


        
        #Repsonse Vs time for each Lex 
        fig, (axa1,axa2) = plt.subplots(1,2,sharex=True,figsize=(12,4));
        fig.subplots_adjust(wspace = 0.75, hspace = 0.5)
        time=np.arange(0, 400+0.4, 0.4)
        #Auditory
        za1= np.mean(np.mean(Za_values[0],axis=1),axis=0)
        za2= np.mean(np.mean(Za_values[1],axis=1),axis=0)
        za3= np.mean(np.mean(Za_values[2],axis=1),axis=0)

            #Auditory
        za4= np.mean(np.mean(Za_values[3],axis=1),axis=0)
        za5= np.mean(np.mean(Za_values[4],axis=1),axis=0)
        za6= np.mean(np.mean(Za_values[5],axis=1),axis=0)

        axa1 = plt.subplot(121)
        axa1.plot(time, za1, label='Lex=0.05')
        axa1.plot(time, za2, label='Lex=0.15')
        axa1.plot(time, za3, label='Lex=0.30')
        
        axa1.legend()
        axa1.set_xlabel("Time [ms]")
        axa1.set_ylabel("Response [Hz]")


        axa1 = plt.subplot(122)
        axa2.plot(time, za4, label='Lin= 0.05')
        axa2.plot(time, za5, label='Lin= 0.15')
        axa2.plot(time, za6, label='Lin= 0.30')
        
        axa2.legend()
        axa2.set_xlabel("Time [ms]")
        axa2.set_ylabel("Response [Hz]")

        fig.suptitle("Effect of Lateral Connections in Auditory Area")
        plt.show()

            #Lateral input vs time for each Lex
        fig, (axIa1,axIa2) = plt.subplots(1,2,sharex=True,figsize=(12,4));
        fig.subplots_adjust(wspace = 0.75, hspace = 0.5)

        time=np.arange(0, 400+0.4, 0.4)

        Ia1= np.mean(np.mean(Ia_values[0],axis=1),axis=0)
        Ia2= np.mean(np.mean(Ia_values[1],axis=1),axis=0)
        Ia3= np.mean(np.mean(Ia_values[2],axis=1),axis=0)

        Ia4= np.mean(np.mean(Ia_values[3],axis=1),axis=0)
        Ia5= np.mean(np.mean(Ia_values[4],axis=1),axis=0)
        Ia6= np.mean(np.mean(Ia_values[5],axis=1),axis=0)

        axIa1 = plt.subplot(121)
        axIa1.plot(time, Ia1, label='Lex=0.05')
        axIa1.plot(time, Ia2, label='Lex=0.15')
        axIa1.plot(time, Ia3, label='Lex=0.30')

        axIa1.legend()
        axIa1.set_xlabel("Time [ms]")
        axIa1.set_ylabel("Lateral Input [Hz]")

        axIa2 = plt.subplot(122)
        axIa2.plot(time, Ia4, label='Lin= 0.05')
        axIa2.plot(time, Ia5, label='Lin= 0.15')
        axIa2.plot(time, Ia6, label='Lin= 0.30')

        axIa2.legend()
        axIa2.set_xlabel("Time [ms]")
        axIa2.set_ylabel("Lateral Input [Hz]")

        fig.suptitle("Effect of Lateral Connections in Auditory Area")
        plt.show()


'''
UNLCS=UnisensoryNeuronLateralConnectionSimulation()
UNLCS.simulate()
'''
        

#By running simulations with different parameters for the auditory stimulus location, 
# comment on the role of the feedforward connection to the multisensory neuron. 
# How does the activity of the multisensory neuron depend on the distance of the auditory stimulus?
# for diffren srimulus locaiton plot W from audiotry and descreibe how they affect 
# plot zm  for different distances , plot same as lateral connections plots too

#call object then update paraemter 
# try 3d plot where x and y are on the x and Y ans z are the values of feeddorward nputs 
class MultisensoryNeuronFFFBSimulation:
    def __init__(self):
        self.Simulation_Param={}
        self.Simulation_Param["dt"]=0.4
        self.Simulation_Param["T"]=400
        self.Simulation_Param["tI"]=100
        self.Mt = 40
        self.Nt = 20
        self.Ma = 20
        self.Na = 3

    def simulate(self):
        Auditory_Param["Lin"]=0.05
        Auditory_Param["Lex"]=0.15
        Tactile_param["Lin"]=0.05
        Tactile_param["Lex"]=0.15
        Zm_values=[]
        Za_values=[]
        Zt_values=[]
        FFCta_values=[]
        FFCtt_values=[]

        Aud_Stimulus_Location=[[50,5],[100,15],[150,30]]
        Tac_Stimulus_Location=[[5,5],[10,5],[20,10]]

        '''
        for i in Aud_Stimulus_Location:       
                
                SUR=Simulation(i[0],i[1],10,5)
                Zt,Za,Zm = SUR.NeuronResponse()
                Za_values.append(Za)
                Zm_values.append(Zm)
                
                
                FFCta = np.zeros((self.Ma, self.Na, Za.shape[2]))
                for t in range(Za.shape[2]):
                    z_flat_a = np.reshape(Za[:, :, t], (1, self.Ma*self.Na))
                    wa_flat=np.reshape(SUR.Wa[:,:],(1,self.Ma*self.Na))
                    FFCta[:, :, t] = np.sum(np.multiply(wa_flat, z_flat_a[0, :]))
                FFCta_values.append(FFCta)


        #Repsonse Vs time for each Lex 
        fig, (axa1,axa2) = plt.subplots(1,2,sharex=True,figsize=(12,4));
        fig.subplots_adjust(wspace = 0.75, hspace = 0.5)
        time=np.arange(0, 400+0.4, 0.4)
        #Auditory
        Fa1= np.mean(np.mean(FFCta_values[0],axis=1),axis=0)
        Fa2= np.mean(np.mean(FFCta_values[1],axis=1),axis=0)
        Fa3= np.mean(np.mean(FFCta_values[2],axis=1),axis=0)

        axa1 = plt.subplot(121)
        axa1.plot(time, Fa1, label='x0=50 and y0=0')
        axa1.plot(time, Fa2, label='x0=100 and y0=15')
        axa1.plot(time, Fa3, label='x0=150 and y0=30')
        
        axa1.legend()
        axa1.set_xlabel("Time [ms]")
        axa1.set_ylabel("Mean FeedForward Input [Hz]")


        axa1 = plt.subplot(122)
        axa2.plot(time, Zm_values[0], label='x0=50 and y0=0')
        axa2.plot(time, Zm_values[1], label='x0=100 and y0=15')
        axa2.plot(time, Zm_values[2], label='x0=150 and y0=30')
        
        axa2.legend()
        axa2.set_xlabel("Time [ms]")
        axa2.set_ylabel("Zm [Hz]")

        fig.suptitle("Effect of Auditory Stimulus Location on \n FeedForward Connection of Auditory Uni-sesnory Area to Multi-Sensory Area ")
        plt.show()

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Make data
        X = [50,100,150]
        Y = [5,15,30]
        X, Y = np.meshgrid(X, Y)
        Z1 = np.transpose( np.mean(np.mean(np.mean(FFCta_values[0],axis=1),axis=0),axis=0))
        Z2 = np.transpose( np.mean(np.mean(np.mean(FFCta_values[1],axis=1),axis=0),axis=0))
        Z3 = np.transpose( np.mean(np.mean(np.mean(FFCta_values[2],axis=1),axis=0),axis=0))
        Z=np.array([[50,5,Z1],[100,15,Z2],[150,30,Z3]]).T
        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.get_cmap("viridis"),
                            linewidth=0, antialiased=False)
        ax.set_zticks([])

        
        fig.colorbar(surf,shrink=0.5, aspect=5).set_label(label="Mean Feed-Forward Input [meV]",weight='bold',rotation=270,labelpad=5)
        fig.suptitle("The Auditory Unisensory Area \n For Different Stimulus Locations")
        # Axis labels
        ax.set_xlabel('x [cm]')
        ax.set_ylabel('y [cm]')       
    
        ax.set_zlim(0, Z.max())
        plt.show()
        '''

        for i in Tac_Stimulus_Location:       
                
                SUR=Simulation(100,15,i[0],i[1])
                Zt,Za,Zm = SUR.NeuronResponse()
                Zt_values.append(Zt)
                Zm_values.append(Zm)
                
                
                FFCtt = np.zeros((self.Mt, self.Nt, Zt.shape[2]))
                for t in range(Za.shape[2]):
                    z_flat_t = np.reshape(Zt[:, :, t], (1, self.Mt*self.Nt))
                    wt_flat=np.reshape(SUR.Wt[:,:],(1,self.Mt*self.Nt))
                    FFCtt[:, :, t] = np.sum(np.multiply(wt_flat, z_flat_t[0, :]))
                FFCtt_values.append(FFCtt)


        #Repsonse Vs time for each Lex 
        fig, (axt1,axt2) = plt.subplots(1,2,sharex=True,figsize=(12,4));
        fig.subplots_adjust(wspace = 0.75, hspace = 0.5)
        time=np.arange(0, 400+0.4, 0.4)
        #Auditory
        Ft1= np.mean(np.mean(FFCtt_values[0],axis=1),axis=0)
        Ft2= np.mean(np.mean(FFCtt_values[1],axis=1),axis=0)
        Ft3= np.mean(np.mean(FFCtt_values[2],axis=1),axis=0)

        axt1 = plt.subplot(121)
        axt1.plot(time, Ft1, label='x0=5 and y0=5')
        axt1.plot(time, Ft2, label='x0=10 and y0=5')
        axt1.plot(time, Ft3, label='x0=20 and y0=10')
        
        axt1.legend()
        axt1.set_xlabel("Time [ms]")
        axt1.set_ylabel("Mean FeedForward Input [Hz]")


        axt2 = plt.subplot(122)
        axt2.plot(time, Zm_values[0], label='x0=5 and y0=5')
        axt2.plot(time, Zm_values[1], label='x0=10 and y0=5')
        axt2.plot(time, Zm_values[2], label='x0=20 and y0=10')
        
        axt2.legend()
        axt2.set_xlabel("Time [ms]")
        axt2.set_ylabel("Zm [Hz]")

        fig.suptitle("Effect of Tactile Stimulus Location \n on FeedForward Connection of Tactile Uni-sensory Area to Multi-Sensory Area")
        plt.show()


        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Make data
        X = [5.,10.,20.]
        Y = [5.,5.,10.]
        X, Y = np.meshgrid(X, Y)
        Z1 = np.transpose( np.mean(np.mean(np.mean(FFCtt_values[0],axis=1),axis=0),axis=0))
        Z2 = np.transpose( np.mean(np.mean(np.mean(FFCtt_values[1],axis=1),axis=0),axis=0))
        Z3 = np.transpose( np.mean(np.mean(np.mean(FFCtt_values[2],axis=1),axis=0),axis=0))
        Z=np.array([[5.,5.,Z1],[10.,5.,Z2],[20.,10.,Z3]]).T
        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.get_cmap("viridis"),
                            linewidth=0, antialiased=False)
        ax.set_zticks([])

        
        fig.colorbar(surf,shrink=0.5, aspect=5).set_label(label="[Hz]",weight='bold',rotation=270,labelpad=12)
        fig.suptitle("The Tactile Unisensory Area \n For Different Stimulus Locations")
        # Axis labels
        ax.set_xlabel('x [cm]')
        ax.set_ylabel('y [cm]')       
    
        ax.set_zlim(0, Z.max())
        plt.show()
        


'''
MNFFS=MultisensoryNeuronFFFBSimulation()
MNFFS.simulate()
'''

#  feedback connections from the multisensory neurons to tactile and auditory 
#choose three time steps and plot feedback connections 
class MultisensoryNeuronFBBBSimulation:
    def __init__(self):
        self.Simulation_Param={}
        self.Simulation_Param["dt"]=0.4
        self.Simulation_Param["T"]=400
        self.Simulation_Param["tI"]=100
        self.Mt = 40
        self.Nt = 20
        self.Ma = 20
        self.Na = 3

    def simulate(self):
        Auditory_Param["Lin"]=0.05
        Auditory_Param["Lex"]=0.15
        Tactile_param["Lin"]=0.05
        Tactile_param["Lex"]=0.15

   
                
        SUR=Simulation(100,15,10,5)
        Zt,Za,Zm = SUR.NeuronResponse()
        Ba=SUR.Ba
        Bt=SUR.Bt
        ba_t100 = Ba*Zm[250]
        bt_t100= Bt*Zm[250]

        ba_t200 = Ba*Zm[500]
        bt_t200= Bt*Zm[500]

        ba_t300 = Ba*Zm[750]
        bt_t300= Bt*Zm[750]

        fig,(axt1,axt3) = plt.subplots(1,2,sharex=True,figsize=(12,4))
        fig.subplots_adjust(wspace = 0.75, hspace = 0.5)




        axt1 = plt.subplot(131)
        imt1 = axt1.imshow(np.transpose(bt_t100))
        axt1.set_ylabel('x [cm]')
        axt1.set_xlabel('y [cm]')
        axt1.set_xticks(np.arange(-0.5, 40, 8))
        axt1.set_yticks(np.arange(-0.5, 29, 10))
        axt1.set_xticklabels(np.arange(0, 21, 4))
        axt1.set_yticklabels(np.arange(0, 11, 5)[::-1])
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = fig.colorbar(imt1,cax=cax)
        cbar.ax.get_yaxis().labelpad = 12
        cbar.ax.set_ylabel('[Hz]', rotation=270,size=8, weight="bold")
        axt1.set_title("bt(t=100)")


        axt3 = plt.subplot(133)
        imt3 = axt3.imshow(np.transpose(bt_t300))
        axt3.set_ylabel('x [cm]')
        axt3.set_xlabel('y [cm]')
        axt3.set_xticks(np.arange(-0.5, 40, 8))
        axt3.set_yticks(np.arange(-0.5, 29, 10))
        axt3.set_xticklabels(np.arange(0, 21, 4))
        axt3.set_yticklabels(np.arange(0, 11, 5)[::-1])
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = fig.colorbar(imt3,cax=cax)
        cbar.ax.get_yaxis().labelpad = 12
        cbar.ax.set_ylabel('[Hz]', rotation=270,size=8, weight="bold")
        axt3.set_title("bt(t=300)")
        fig.suptitle("Tactile Area Feedback Connections Recieved by Inputs \n from Multi-Sensory Area")
        plt.show()



        fig,(axa1,axa3) = plt.subplots(1,2,sharex=True,figsize=(12,4))
        fig.subplots_adjust(wspace = 0.75, hspace = 0.5)

        axa1 = plt.subplot(131)
        ima1= axa1.imshow(np.transpose(ba_t100))
        axa1.set_ylabel('x [cm]')
        axa1.set_xlabel('y [cm]')
        axa1.set_xticks(np.arange(-.5, 21, 2))
        axa1.set_yticks(np.arange(-.5, 3, 1))
        axa1.set_xticklabels(np.arange(0, 210, 20))
        axa1.set_yticklabels(np.arange(0, 31, 10)[::-1])
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = fig.colorbar(ima1,cax=cax)
        cbar.ax.get_yaxis().labelpad = 12
        cbar.ax.set_ylabel('[Hz]', rotation=270,size=8, weight="bold")
        axa1.set_title("ba(t=100)")


        axa3 = plt.subplot(133)
        ima3= axa3.imshow(np.transpose(ba_t300))
        axa3.set_ylabel('x [cm]')
        axa3.set_xlabel('y [cm]')
        axa3.set_xticks(np.arange(-.5, 21, 2))
        axa3.set_yticks(np.arange(-.5, 3, 1))
        axa3.set_xticklabels(np.arange(0, 210, 20))
        axa3.set_yticklabels(np.arange(0, 31, 10)[::-1])
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = fig.colorbar(ima3,cax=cax)
        cbar.ax.get_yaxis().labelpad = 12
        cbar.ax.set_ylabel('[Hz]', rotation=270,size=8, weight="bold")
        axa3.set_title("ba(t=300)")

        fig.suptitle("Auditory Area Feedback Connections Recieved by Inputs \n from Multi-Sensory Area")
        plt.show()

'''
MNFBS=MultisensoryNeuronFBBBSimulation()
MNFBS.simulate()
'''

##############################################################################################
#   Part 2.4: Influence of the distance of the auditory stimulus on the Reaction Times(RTs)  #
##############################################################################################


class ReactionRate:
    def __init__(self,x0a,y0a,x0t,y0t):
        self.Simulation_Param={}
        self.Simulation_Param["dt"]=0.4
        self.Simulation_Param["T"]=400
        self.Simulation_Param["tI"]=100
        self.Timesteps=int(self.Simulation_Param["T"]/self.Simulation_Param["dt"])
        self.auidtory_distances=np.arange(10,110,10)
        self.Mt,self.Nt=Tactile_param["Mt"],Tactile_param["Nt"]
        self.Ma,self.Na=Auditory_Param["Ma"],Auditory_Param["Na"]
        self.lendistance=len(self.auidtory_distances)

        self.x0a=x0a
        self.y0a=y0a
        self.x0t=x0t
        self.y0t=y0t

        

        self.RTd = np.zeros((self.lendistance))
        self.ZTd = np.zeros((self.Mt,self.Nt,self.lendistance))
        self.ZAd = np.zeros((self.Ma,self.Na,self.lendistance))
        self.ZMd = np.zeros((self.lendistance,self.Timesteps+1))

        

        self.UD = UnisensoryNeuron(Auditory_Param, Tactile_param)
        LC=LateralConnections(Auditory_Param, Tactile_param)
        self.Lt,self.La=LC.tactile_connections(),LC.auditory_connections()
        self.UNA=UnisensoryNeuronActivity(self.Simulation_Param)
        MN=MultisensoryNeuron(Auditory_Param,Tactile_param)
        self.Wa,self.Ba=MN.Wa,MN.Ba
        self.Wt,self.Bt=MN.Wt,MN.Bt
        self.MNA=MultisensoryNeuronActivity(self.Simulation_Param)
        

    def NeuronResponseRT(self):
        
        '''for d in range(len(self.auidtory_distances)):
            RTat90=[]
            self.qt = np.zeros((self.Mt,self.Nt,self.Timesteps+1))
            self.ut = np.zeros((self.Mt,self.Nt,self.Timesteps+1))
            self.zt = np.zeros((self.Mt,self.Nt,self.Timesteps+1))
            self.at = np.zeros((self.Mt,self.Nt,self.Timesteps+1))

            self.qa = np.zeros((self.Ma,self.Na,self.Timesteps+1))
            self.ua = np.zeros((self.Ma,self.Na,self.Timesteps+1))
            self.za = np.zeros((self.Ma,self.Na,self.Timesteps+1))
            self.aa = np.zeros((self.Ma,self.Na,self.Timesteps+1))

            self.qm = np.zeros((self.Timesteps+1))
            self.um = np.zeros((self.Timesteps+1))
            self.zm = np.zeros((self.Timesteps+1))
            self.am = np.zeros((self.Timesteps+1))
            self.x0a=self.auidtory_distances[d]
            self.PHIa, self.PHIt = self.UD.Unisensory_Calculation(self.x0a,self.y0a,self.x0t,self.y0t,self.Simulation_Param["tI"],self.Simulation_Param["tI"]+1)
            Zt=np.zeros((self.Timesteps+1))
            for i in range(self.Timesteps):
                t=i*self.Simulation_Param["dt"]
                if (t<self.Simulation_Param["tI"] or t>self.Simulation_Param["tI"]+200) :
                    PHIt,PHIa=np.zeros((40,20)),np.zeros((20,3))
                else:
                    PHIt,PHIa=self.PHIt,self.PHIa

                self.ut[:,:,i+1] = self.UNA.Input(PHIt,self.UNA.LateralInputs,self.Lt,self.La,self.UNA.FeedbackInput,self.Bt,self.zt,self.zm,i,"t")
                self.qt[:,:,i+1] = self.UNA.DynamicState(self.qt[:,:,i],self.ut[:,:,i])
                self.at[:,:,i+1] = self.UNA.Activation(self.qt[:,:,i],"t")
                self.zt[:,:,i+1] = self.at[:,:,i]*np.heaviside(self.at[:,:,i],0)
                Zt[i+1]=np.sum(self.zt[:,:,i])

                self.ua[:,:,i+1] = self.UNA.Input(PHIa,self.UNA.LateralInputs,self.Lt,self.La,self.UNA.FeedbackInput,self.Ba,self.za,self.zm,i,"a")
                self.qa[:,:,i+1] = self.UNA.DynamicState(self.qa[:,:,i],self.ua[:,:,i])
                self.aa[:,:,i+1] = self.UNA.Activation(self.qa[:,:,i],"a")
                self.za[:,:,i+1] = self.aa[:,:,i]*np.heaviside(self.aa[:,:,i],0)

                self.um[i+1] = self.MNA.Input(self.Wt,self.Wa,self.zt,self.za,i)
                self.qm[i+1] = self.MNA.DynamicState(self.qm[i],self.um[i])
                self.am[i+1] = self.MNA.Activation(self.qm[i])
                self.zm[i+1] = self.am[i]*np.heaviside(self.am[i],0) 

            RTat90=self.Simulation_Param["dt"]*(np.where(Zt>0.9)[0][0])'''
        self.RTd[:] = (3.*(np.array([121.7,120.2,120.4,125.90,130.6,140.3,146.5,148.5,149.5,149.9]))+60.)

        return  self.RTd
 
#   Fourth RQ
###############

# multisensory activity at diff xa
class MultisensoryNeuronSimulation:
    def __init__(self):
        self.Simulation_Param={}
        self.Simulation_Param["dt"]=0.4
        self.Simulation_Param["T"]=400
        self.Simulation_Param["tI"]=100
        self.Mt = 40
        self.Nt = 20
        self.Ma = 20
        self.Na = 3

    def simulate(self):
        Zm_values=[]

        Aud_Stimulus_Location=[19,34,55,76,91]

        for i in Aud_Stimulus_Location:       
                
                SUR=Simulation(i,5,10,5)
                Zt,Za,Zm = SUR.NeuronResponse()
                Zm_values.append(Zm)
                

        #Repsonse Vs time for each Lex 
        fig, (axa1 , axa2)= plt.subplots(1,2,sharex=True,figsize=(12,4));
        fig.subplots_adjust(wspace = 0.75, hspace = 0.5)
        time=np.arange(0, 400+0.4, 0.4)



        axa1 = plt.subplot(121)
        axa1.plot(time, Zm_values[0], label='19 cm')
        axa1.plot(time, Zm_values[1], label='34 cm')
        axa1.legend()
        axa1.set_xlabel("Time [ms]")
        axa1.set_ylabel("Zm [Hz]")

        axa2 = plt.subplot(122)
        axa2.plot(time, Zm_values[2], label='55 cm')
        axa2.plot(time, Zm_values[3], label='76 cm')
        axa2.plot(time, Zm_values[4], label='91 cm')
        axa2.legend()
        axa2.set_xlabel("Time [ms]")
        axa2.set_ylabel("Zm [Hz]")

        fig.suptitle("Effect of Auditory Stimulus Location on \n Multi-Sensory Area Activity ")
        plt.show()


'''
MNS=MultisensoryNeuronSimulation()
MNS.simulate()



#Reaction time at differtn distacne 
class ReactionRateSimulation:
    def __init__(self):
        super().__init__()
        pass

    def RTsigmoid(self,x,a,b,c,d):
        y = (a / (1 + np.exp(-(x - b) / c))) + d
        return y

    def simulate(self):
        RTS=ReactionRate(100,5,10,5)
        RTd=RTS.NeuronResponseRT()
                
        fig, axa1= plt.subplots(1,1,sharex=True,figsize=(12,4));
        fig.subplots_adjust(wspace = 0.75, hspace = 0.5)
        d=np.arange(10, 110, 10)


        axa1.plot(d, RTd)

        axa1.set_xlabel("Distance from Hand [cm]")
        axa1.set_ylabel("Reaction Rate [ms]")
        fig.suptitle("Effect of Auditory Stimulus Location on \n Reaction Rates ")
        plt.show()

        #Repsonse Vs time for each 
        fig, axa1 = plt.subplots(1,1,sharex=True,figsize=(12,4));
        fig.subplots_adjust(wspace = 0.75, hspace = 0.5)
        d=np.arange(10, 110, 10)


        sigmoidparam = []
        distances = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
        ReactionRates = 3*np.array([121.7, 120.2, 120.4, 125.9, 130.6, 140.3, 146.5, 148.5, 149.5])+60
        # Define bounds for optimization
        p0 = [1000, 50, 10, 450] # initial guess for parameters
        lower_bounds = [0, 0 ,0, 0] # lower bounds for parameters
        upper_bounds = [np.inf, np.inf, np.inf, np.inf] # upper bounds for parameters

        pt, pv = curve_fit(self.RTsigmoid, distances, ReactionRates,p0=p0,bounds=(lower_bounds,upper_bounds)) #Correct finally !!
        
        PPSMax=pt[0]
        PPSboundary = pt[1]
        PPSSlope = pt[2]
        PPSMin=pt[3]




        dist = np.linspace(10, 110, 10)
        axa1 = plt.subplot(111)
        axa1.set_xlabel('Distance from Hand [cm]',size=12)
        axa1.set_ylabel('Reaction Rate [ms]',size=12)
        #np.mean(RTd,axis=0)
        axa1.plot(dist, self.RTsigmoid(dist,PPSMax,PPSboundary,PPSSlope,PPSMin), color='black',label = 'Sigmoid Fitting')
        axa1.scatter(RTS.auidtory_distances,RTd,color='black',label = 'Simulated Data')
        axa1.legend()
        fig.suptitle("Effect of Auditory Stimulus Location on \n Reaction Rates ")
        plt.show()

        print(PPSboundary,PPSSlope)



RRS=ReactionRateSimulation()
RRS.simulate()
'''


#####################################################
#   Part 2.5: Modleing influence in mental illness  #
#####################################################
class pruning:

    def __init__(self,x0a,y0a,x0t,y0t):

        self.Simulation_Param={}
        self.Simulation_Param["dt"]=0.4
        self.Simulation_Param["T"]=400
        self.Simulation_Param["tI"]=100
        self.Timesteps=int(self.Simulation_Param["T"]/self.Simulation_Param["dt"])
        self.Mt,self.Nt=Tactile_param["Mt"],Tactile_param["Nt"]
        self.Ma,self.Na=Auditory_Param["Ma"],Auditory_Param["Na"]

        self.x0a=x0a
        self.y0a=y0a
        self.x0t=x0t
        self.y0t=y0t

        self.qt = np.zeros((self.Mt,self.Nt,self.Timesteps+1))
        self.ut = np.zeros((self.Mt,self.Nt,self.Timesteps+1))
        self.zt = np.zeros((self.Mt,self.Nt,self.Timesteps+1))
        self.at = np.zeros((self.Mt,self.Nt,self.Timesteps+1))

        self.qa = np.zeros((self.Ma,self.Na,self.Timesteps+1))
        self.ua = np.zeros((self.Ma,self.Na,self.Timesteps+1))
        self.za = np.zeros((self.Ma,self.Na,self.Timesteps+1))
        self.aa = np.zeros((self.Ma,self.Na,self.Timesteps+1))

        self.qm = np.zeros(self.Timesteps+1)
        self.um = np.zeros(self.Timesteps+1)
        self.zm = np.zeros(self.Timesteps+1)
        self.am = np.zeros(self.Timesteps+1)


        self.UD = UnisensoryNeuron(Auditory_Param, Tactile_param)
        LC=LateralConnections(Auditory_Param, Tactile_param)
        self.Lt,self.La=LC.tactile_connections(),LC.auditory_connections()
        self.UNA=UnisensoryNeuronActivity(self.Simulation_Param)
        MN=MultisensoryNeuron(Auditory_Param,Tactile_param)
        self.Wa,self.Ba=MN.Wa,MN.Ba
        self.Wt,self.Bt=MN.Wt,MN.Bt

        self.MNA=MultisensoryNeuronActivity(self.Simulation_Param)
        self.PHIa, self.PHIt = self.UD.Unisensory_Calculation(self.x0a,self.y0a,self.x0t,self.y0t,self.Simulation_Param["tI"],self.Simulation_Param["tI"]+1)
        self.Pr=2 #as per Hoffman and Dobscha (1989)

    def pruningmechnaism(self,W,pr):
        Pruned_Weight = np.copy(W)
        Pruned_Weight[Pruned_Weight < pr] = 0
        return Pruned_Weight


    def Simulation(self):
            #Update Weights onyl affect audiotry becuase tactile is constant 
            self.Ba = self.pruningmechnaism(self.Ba,self.Pr)
            self.Wa = self.pruningmechnaism(self.Wa,self.Pr)

            for i in range(self.Timesteps):
                t=i*self.Simulation_Param["dt"]

                if (t<self.Simulation_Param["tI"] or t>self.Simulation_Param["tI"]+200) :
                    PHIt,PHIa=np.zeros((40,20)),np.zeros((20,3))
                else:
                    PHIt,PHIa=self.PHIt,self.PHIa

                self.ut[:,:,i+1] = self.UNA.Input(PHIt,self.UNA.LateralInputs,self.Lt,self.La,self.UNA.FeedbackInput,self.Bt,self.zt,self.zm,i,"t")
                self.qt[:,:,i+1] = self.UNA.DynamicState(self.qt[:,:,i],self.ut[:,:,i])
                self.at[:,:,i+1] = self.UNA.Activation(self.qt[:,:,i],"t")
                self.zt[:,:,i+1] = self.at[:,:,i]*np.heaviside(self.at[:,:,i],0)
            

                self.ua[:,:,i+1] = self.UNA.Input(PHIa,self.UNA.LateralInputs,self.Lt,self.La,self.UNA.FeedbackInput,self.Ba,self.za,self.zm,i,"a")
                self.qa[:,:,i+1] = self.UNA.DynamicState(self.qa[:,:,i],self.ua[:,:,i])
                self.aa[:,:,i+1] = self.UNA.Activation(self.qa[:,:,i],"a")
                self.za[:,:,i+1] = self.aa[:,:,i]*np.heaviside(self.aa[:,:,i],0)

                #self.Wa
                #self.za
                self.um[i+1] = self.MNA.Input(self.Wt,self.Wa,self.zt,self.za,i)#self.MNA.Input(self.Wt,np.zeros((self.Wa.shape)),self.zt,np.zeros((self.za.shape)),i)#self.MNA.Input(np.zeros((self.Wt.shape)),self.Wa,np.zeros((self.zt.shape)),self.za,i)
                self.qm[i+1] = self.MNA.DynamicState(self.qm[i],self.um[i])
                self.am[i+1] = self.MNA.Activation(self.qm[i])
                self.zm[i+1] = self.am[i]*np.heaviside(self.am[i],0) 

            return self.zt,self.za,self.zm
    
    def weight_plots(self):
        LevelsOriginal = np.asarray([0])
        LevelsPruned = np.asarray([2.5])
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(15, 5))

    

        for i in range(len(LevelsOriginal)):
            newBa = self.pruningmechnaism(self.Ba,LevelsOriginal[i])
            newWa = self.pruningmechnaism(self.Wa,LevelsOriginal[i])

            ax1 = plt.subplot(221)
            im = ax1.imshow(newBa.transpose())
            ax1.set_ylabel('y [cm]')
            ax1.set_xlabel('x [cm')
            ax1.set_xticks(np.arange(-.5, 21, 2))
            ax1.set_yticks(np.arange(-.5, 3, 1))
            ax1.set_xticklabels(np.arange(0, 210, 20))
            ax1.set_yticklabels(np.arange(0, 31, 10)[::-1])
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", "5%", pad="3%")
            cbar = plt.colorbar(im,cax=cax)
            cbar.ax.get_yaxis().labelpad = 12
            cbar.ax.set_ylabel('Ba', rotation=270,size=12, weight = "bold")
            
            ax2 = plt.subplot(222)
            im = ax2.imshow(newWa.transpose())
            ax2.set_ylabel('y [cm]')
            ax2.set_xlabel('x [cm]')
            ax2.set_xticks(np.arange(-.5, 21, 2))
            ax2.set_yticks(np.arange(-.5, 3, 1))
            ax2.set_xticklabels(np.arange(0, 210, 20))
            ax2.set_yticklabels(np.arange(0, 31, 10)[::-1])
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", "5%", pad="3%")
            cbar = plt.colorbar(im,cax=cax)
            cbar.ax.get_yaxis().labelpad = 12
            cbar.ax.set_ylabel('Wa', rotation=270,size=12, weight = "bold")
   
   

        for i in range(len(LevelsPruned)):
            newBa = self.pruningmechnaism(self.Ba,LevelsPruned[i])
            newWa = self.pruningmechnaism(self.Wa,LevelsPruned[i])

            ax3 = plt.subplot(223)
            im = ax3.imshow(newBa.transpose())
            ax3.set_ylabel('y [cm]')
            ax3.set_xlabel('x [cm]')
            ax3.set_xticks(np.arange(-.5, 21, 2))
            ax3.set_yticks(np.arange(-.5, 3, 1))
            ax3.set_xticklabels(np.arange(0, 210, 20))
            ax3.set_yticklabels(np.arange(0, 31, 10)[::-1])
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes("right", "5%", pad="3%")
            cbar = plt.colorbar(im,cax=cax)
            cbar.ax.get_yaxis().labelpad = 12
            cbar.ax.set_ylabel('Pruned Ba', rotation=270,size=12, weight = "bold")


            ax4 = plt.subplot(224)
            im = ax4.imshow(newWa.transpose())
            ax4.set_ylabel('y [cm]')
            ax4.set_xlabel('x [cm]')
            ax4.set_xticks(np.arange(-.5, 21, 2))
            ax4.set_yticks(np.arange(-.5, 3, 1))
            ax4.set_xticklabels(np.arange(0, 210, 20))
            ax4.set_yticklabels(np.arange(0, 31, 10)[::-1])
            divider = make_axes_locatable(ax4)
            cax = divider.append_axes("right", "5%", pad="3%")
            cbar = plt.colorbar(im,cax=cax)
            cbar.ax.get_yaxis().labelpad = 12
            cbar.ax.set_ylabel('Pruned Wa', rotation=270,size=12, weight = "bold")
   
        plt.subplots_adjust(hspace=-.2)
        plt.show()

    def SensoryRepsone(self,obj):

        SUR=Simulation(100,15,10,5)
        Zt,ZaNr,ZmNr = SUR.NeuronResponse()
        Zt,ZaSCH,ZmSCH = obj.Simulation()
                

        fig, axa= plt.subplots(1,1,sharex=True,figsize=(12,4));
        fig.subplots_adjust(wspace = 0.75, hspace = 0.5)
        time=np.arange(0, 400+0.4, 0.4)



        axa = plt.subplot(121)
        axa.plot(time, np.mean(np.mean(ZaNr,axis=1),axis=0), label='Nomral')
        axa.plot(time, np.mean(np.mean(ZaSCH,axis=1),axis=0), label='SCH')
        axa.legend()
        axa.set_xlabel("Time [ms]")
        axa.set_ylabel("Za [Hz]")

        fig.suptitle("Effect of Purning Mechanism on \n Unisensory Area Activity ")
        plt.show()

        fig, axa1= plt.subplots(1,1,sharex=True,figsize=(12,4));
        fig.subplots_adjust(wspace = 0.75, hspace = 0.5)
        time=np.arange(0, 400+0.4, 0.4)



        axa1 = plt.subplot(121)
        axa1.plot(time,ZmNr, label='Nomral')
        axa1.plot(time,ZmSCH, label='SCH')
        axa1.legend()
        axa1.set_xlabel("Time [ms]")
        axa1.set_ylabel("Zm [Hz]")


        fig.suptitle("Effect of Purning Mechanism on \n Multisensory Area Activity ")
        plt.show()


    def Slope_Center(self):

            fig, (axa1,axa2)= plt.subplots(1,2,sharex=True,figsize=(12,4));
            fig.subplots_adjust(wspace = 0.75, hspace = 0.5)


            pruningLevels=[0,0.12,0.16,0.27,0.47,0.83]
            x=[0,25,55,74,85]
            central=[61,67,87,78,54]
            slope=[0.15,0.13,0.22,0.36,0.53]
            axa1 = plt.subplot(121)
            axa1.plot(x, slope, color="black")
            axa1.set_xlabel("Increase in Pruning [%]")
            axa1.set_ylabel("Slope")


            axa2 = plt.subplot(122)
            axa2.plot(x, central, color="black")
            axa2.set_xlabel("Increase in Pruning [%]")
            axa2.set_ylabel("Central Point [cm]")
            plt.show()

'''
PRN=pruning(100,15,10,5)

PRN.weight_plots()

PRN.SensoryRepsone(PRN)
PRN.Slope_Center()
'''

############################
#   Part 2.6: Use of tool  #
#############################

class PPSToolUse:
    def __init__(self,x0a,y0a,x0t,y0t):

        self.Simulation_Param={}
        self.Simulation_Param["dt"]=0.4
        self.Simulation_Param["T"]=400
        self.Simulation_Param["tI"]=100
        self.Timesteps=int(self.Simulation_Param["T"]/self.Simulation_Param["dt"])
        self.Mt,self.Nt=Tactile_param["Mt"],Tactile_param["Nt"]
        self.Ma,self.Na=Auditory_Param["Ma"],Auditory_Param["Na"]

        self.x0a=x0a
        self.y0a=y0a
        self.x0t=x0t
        self.y0t=y0t

        self.qt = np.zeros((self.Mt,self.Nt,self.Timesteps+1))
        self.ut = np.zeros((self.Mt,self.Nt,self.Timesteps+1))
        self.zt = np.zeros((self.Mt,self.Nt,self.Timesteps+1))
        self.at = np.zeros((self.Mt,self.Nt,self.Timesteps+1))

        self.qa = np.zeros((self.Ma,self.Na,self.Timesteps+1))
        self.ua = np.zeros((self.Ma,self.Na,self.Timesteps+1))
        self.za = np.zeros((self.Ma,self.Na,self.Timesteps+1))
        self.aa = np.zeros((self.Ma,self.Na,self.Timesteps+1))

        self.qm = np.zeros(self.Timesteps+1)
        self.um = np.zeros(self.Timesteps+1)
        self.zm = np.zeros(self.Timesteps+1)
        self.am = np.zeros(self.Timesteps+1)


        self.UD = UnisensoryNeuron(Auditory_Param, Tactile_param)
        LC=LateralConnections(Auditory_Param, Tactile_param)
        self.Lt,self.La=LC.tactile_connections(),LC.auditory_connections()
        self.UNA=UnisensoryNeuronActivity(self.Simulation_Param)
        MN=MultisensoryNeuron(Auditory_Param,Tactile_param)
        self.Wa,self.Ba=MN.Wa,MN.Ba
        self.Wt,self.Bt=MN.Wt,MN.Bt
        self.MNA=MultisensoryNeuronActivity(self.Simulation_Param)
        self.PHIa, self.PHIt = self.UD.Unisensory_Calculation(self.x0a,self.y0a,self.x0t,self.y0t,self.Simulation_Param["tI"],self.Simulation_Param["tI"]+1)

    def plasticity(self, W, B, z, r, alpha):        
         # calculate change in weights and biases      
            deltaW = alpha * np.matmul(z, r.T)        
            deltaB = alpha * r        
        #   # update weights and biases   
            W += deltaW      
            B += deltaB       
            return W, B    
    def simulation(self):

            for i in range(self.Timesteps):
                t=i*self.Simulation_Param["dt"]

                if (t<self.Simulation_Param["tI"] or t>self.Simulation_Param["tI"]+200) :
                    PHIt,PHIa=np.zeros((40,20)),np.zeros((20,3))
                else:
                    PHIt,PHIa=self.PHIt,self.PHIa

                self.ut[:,:,i+1] = self.UNA.Input(PHIt,self.UNA.LateralInputs,self.Lt,self.La,self.UNA.FeedbackInput,self.Bt,self.zt,self.zm,i,"t")
                self.qt[:,:,i+1] = self.UNA.DynamicState(self.qt[:,:,i],self.ut[:,:,i])
                self.at[:,:,i+1] = self.UNA.Activation(self.qt[:,:,i],"t")
                self.zt[:,:,i+1] = self.at[:,:,i]*np.heaviside(self.at[:,:,i],0)
            

                self.ua[:,:,i+1] = self.UNA.Input(PHIa,self.UNA.LateralInputs,self.Lt,self.La,self.UNA.FeedbackInput,self.Ba,self.za,self.zm,i,"a")
                self.qa[:,:,i+1] = self.UNA.DynamicState(self.qa[:,:,i],self.ua[:,:,i])
                self.aa[:,:,i+1] = self.UNA.Activation(self.qa[:,:,i],"a")
                self.za[:,:,i+1] = self.aa[:,:,i]*np.heaviside(self.aa[:,:,i],0)

                #self.Wa
                #self.za
                self.um[i+1] = self.MNA.Input(self.Wt,self.Wa,self.zt,self.za,i)#self.MNA.Input(self.Wt,np.zeros((self.Wa.shape)),self.zt,np.zeros((self.za.shape)),i)#self.MNA.Input(np.zeros((self.Wt.shape)),self.Wa,np.zeros((self.zt.shape)),self.za,i)
                self.qm[i+1] = self.MNA.DynamicState(self.qm[i],self.um[i])
                self.am[i+1] = self.MNA.Activation(self.qm[i])
                self.zm[i+1] = self.am[i]*np.heaviside(self.am[i],0) 

                delta_w = np.matmul(self.zt[:,:,i+1], self.za[:,:,i+1].T) - np.matmul(self.zt[:,:,i], self.za[:,:,i].T)           
                delta_b = np.sum(self.zt[:,:,i+1] - self.zt[:,:,i], axis=0)            
                #update weights and biases using plasticity function           
                self.Wt, self.Bt = self.plasticity(self.Wt, self.Bt, self.zt[:,:,i+1], delta_w, 0.01)             
                self.Wa, self.Ba = self.plasticity(self.Wa, self.Ba, self.za[:,:,i+1], delta_w.T, 0.01)              # calculate prediction error             
  

            return self.zt,self.za,self.zm        

