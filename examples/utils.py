import numpy as np
import matplotlib.pyplot as plt 

def BfreqS(G, flag):
    """
    For SISO systems in the transfer function form:

    ┌──────┬────────────────┐
    │ Flag │ Plot type      │
    ├──────┼────────────────┤
    │  1   │ Bode           │
    │  2   │ Bode magnitude │
    │  3   │ Nyquist        │
    │  4   │ Nichols        │
    │  5   │ ALL            │
    └──────┴────────────────┘ 
    """

    if hasattr(G, 'name'):
        title_suffix = f" for {G.name}"
    else:
        title_suffix = ""
    
    num         = G.num[0][0]
    den         = G.den[0][0]
    zeros       = np.roots(num)
    poles       = np.roots(den)
    
    pi          = poles[np.abs(np.real(poles))<1e-14]
    dmy         = np.append(np.abs(zeros), np.abs(poles))
    rinf        = np.max([100*np.max(dmy),100])
    reps        = rinf/1e3
    w2          = rinf
    znz         = zeros[np.abs(zeros)>1e-6]
    pnz         = poles[np.abs(poles)>1e-6]
    dmy         = np.append(np.abs(znz),np.abs(pnz))
    w1          = (1/20)*np.min(dmy)
    w           = np.logspace(np.log10(w1),np.log10(w2),num=2000)   #Bode freqs
    w2          = np.logspace(np.log10(w1/20),np.log10(w2*20),num=2000) 

    pi          = np.imag(pi)
    pi          = np.sort(pi)

    tet         = np.pi*np.arange(180,-180,-1)/360
    cwhc        = np.exp(1j*tet)
    tet         = np.pi*np.arange(-180,180,-1)/360
    ccwhc       = np.exp(1j*tet)

    if np.size(pi) > 0:
        z1          = -w[::-1]
        dmy         = 1j*np.append(z1,w)
        z           = np.append(dmy,rinf*cwhc)
    else:
        z           = 1j*np.arange(-rinf,(pi[0]-reps))
        z           = np.append(z,j*pi[0]+reps*ccwhc)
        if np.size(pi)>1:
            for k in range(1,np.size(pi)):
                drs     = (pi[k]-pi[k-1]-2*reps)/200
                zk1     = 1j*np.arange(pi[k-1]+reps,pi[k]-reps,drs)
                zk2     = 1j*pi[k]+reps*ccwhc 
                zk      = np.append(zk1,zk2)
                z       = np.append(z,zk)
        drs         = (rinf-pi[-1]-reps)/200.
        z           = np.append(z,1j*np.arange(pi[-1]+reps,rinf,drs))    
        z           = np.append(z,rinf*cwhc)
    
    Gw          = np.polyval(num,1j*w)/np.polyval(den,1j*w)
    mag         = 20*np.log10(np.abs(Gw))
    phs         = np.angle(Gw,deg=True)

    Gw2         = np.polyval(num,1j*w2)/np.polyval(den,1j*w2)
    mag2        = 20*np.log10(np.abs(Gw2))
    phs2        = np.angle(Gw2,deg=True)

    Gz          = np.polyval(num,z)/np.polyval(den,z)

    if flag == 5 or flag == 1:
        plt.subplot(211)
        plt.semilogx(w,mag)
        plt.grid()
        plt.subplot(212)
        plt.semilogx(w,np.unwrap(phs))
        plt.grid()
        plt.title(f"Bode plot{title_suffix}")
        plt.show()
    elif flag == 5 or flag == 2:
        plt.semilogx(w,mag)
        plt.grid()
        plt.title(f"Bode magnitude plot{title_suffix}")
        plt.show()
    elif flag == 5 or flag == 3:
        plt.plot(np.real(Gz),np.imag(Gz),-1,0,'r+')
        plt.grid()
        plt.title(f"Nyquist plot{title_suffix}")
        plt.show()
    elif flag == 5 or flag == 4:
        bx      = [-150,-150,-210,-210,-150]
        by      = [6,-6,-6,6,6]
        bx2     = [150,150,210,210,150]
        by2     = [6,-6,-6,6,6]
        phsp    = np.unwrap(phs2) 
        if np.max(phsp) < 170:
            plt.plot(phsp,mag2,bx,by,'r--',-180,0,'r+')
        else:
            plt.plot(phsp,mag2,bx,by,'r--',-180,0,'r+',bx2,by2,'r--',180,0,'r+')
        plt.grid()
        plt.title(f"Nichols plot{title_suffix}")
        plt.show()

    return Gz