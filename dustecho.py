#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 13:14:59 2023

@author: tathagata
"""

import numpy as np
import scipy.constants as cns
import scipy.integrate as intg

L0 = 1e44
pc = 3.086e+16
days_to_sec = 3600*24
t0 = 1500*days_to_sec
s0 = 20*days_to_sec

def gauss(t,t0,sd):
    tt = (t - t0)/sd
    ff = np.exp(-tt**2/2) + 1e-5 #add a gaussian flare
    return ff


class dustecho:
    def __init__(self, idx, id1):
        """ Initialize the geometry. """
        self.pc = pc
        self.index = idx
        self.rin = 0.2*pc
        self.rout = 1.0*pc
        self.alpha = 0.1
        self.id1 = id1
        self.s02 = 400*days_to_sec
        if idx == "full-shell":
            self.thetao = 0.
        elif idx == "biconically-cutout-shell":
            print("The default opening angle is 45 degrees. Change it if you want something different.")
            self.thetao = 45.*np.pi/180.
        else :
            print("Invalid geometery.. reset to full-shell..")
            self.thetao = 0.
    
    def set_the_input_lc_params(self, id1):
        """ Set the input light curve parameters """
        if self.id1 == 1:
            self.t01 = t0
            self.sd1 = s0
        elif self.id1 == 2:
            self.t01 = t0
            self.sd1 = s0
            self.t02 = t0
            self.sd2 = self.s02
        else:
            self.id1 = 1
            self.t01 = t0
            self.sd1 = s0
    
    def fr(self, f0, r):
        """RADIAL TRANSPORT"""
        d0 = (r-self.rin)/(self.pc)
        tau = self.alpha*d0
        L = np.exp(-tau)*f0
        return L

    def Rt(self, theta, r):
        """REST OF THE JOURNEY AFTER"""
        ff = np.sqrt( self.rout**2-(r*np.sin(theta))**2 )-r*np.cos(theta)
        return ff
    
    
    def Lin(self, t):
        """ The incident radiation """
        A = 1e44
        ff = A*gauss(t, self.t01, self.sd1) + 1e44
        return ff

    def Lin2(self, t):
        """ The incident radiation """
        amplitude1 = 1e44
        amplitude2 = 5e43
        ff = amplitude1 * gauss(t,self.t01,self.sd1) + amplitude2 * gauss(t,self.t02,self.sd2) + 1e43 
        return ff

    def d(self, r,theta):
        """ Light travel distance """
        ff = r*(1-np.cos(theta))
        return ff

    def Finc1(self,t,R,theta):
        """ Incident flux: should be modified here """
        #F0 = L0/(4*np.pi*rin**2)
        if self.id1 == 1:        
            Finc = self.Lin(t - self.d(R,theta)/cns.c) / (4*np.pi*R**2)
        elif self.id1 == 2:
            Finc = self.Lin2(t - self.d(R,theta)/cns.c) / (4*np.pi*R**2)
        ff0 = self.fr(Finc,R)
        return ff0


    def Finc2(self, t,R):
        """ Incident flux theta integrated """
        dtheta = 0.004 # Theta integration steps for simpson
        theta = np.arange(self.thetao, np.pi/2, dtheta)
        ftr = []
        for i in range(len(theta)):
            ft0 = self.Finc1(t,R,theta[i])
            tau1 = self.alpha*self.Rt(theta[i],R)/self.pc
            frac = np.exp(-tau1)
            ftr.append( ft0*frac)
        ftr = np.array(ftr)    
        ff  = 2*intg.simps( ftr*np.cos(theta)*np.sin(theta), x=theta )
        return ff

    def Finc3(self, t):
        dr   = 0.05*pc
        r = np.arange(self.rin, self.rout, dr, dtype=float)
        Fi = []
        for i in range(r.size):
            Fi.append( self.Finc2(t,r[i]) )
        Fi = np.array(Fi)
        ff = intg.simps(Fi*2*np.pi*r, dx=dr)
        return ff
    
    def f0(self, t):
        flux = []
        for i in range(t.size):
            flux.append( self.Finc3(t[i]) )
        return flux
        