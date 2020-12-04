# Copyright 2020 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Library of synthesizer functions."""

from ddsp import core
from ddsp import processors
import gin
import tensorflow.compat.v2 as tf



@gin.register
class FTM(processors.Processor):
  """Synthesize percussive sounds based on physical parameters."""

  def __init__(self,
               n_samples=2**15,
               sample_rate=16000,
               mode=20,
               name='ftm'):
    super().__init__(name=name) #use super to call processor's init class
    self.n_samples = n_samples
    self.sample_rate = sample_rate
    self.mode = mode
   
   
  ## helper function 
  def approxnorm(self,l,mu,s,tau): #take tensor, float, float, int
    # this is a gaussian distribution
      h = l/tau #tensor
      i = core.tf_float32(tf.range(0,tau+1,1))
      x = 1/(s * tf.math.sqrt(2*np.pi) * tf.math.exp(-0.5 * (i * h - mu)**2/s**2))

      #i = np.arange(0,tau+1,1)
      #x = 1/(s * np.sqrt(2*np.pi)) * np.exp(-0.5 * (i * h - mu)**2/s**2)
    
      return core.tf_float32(x) #return list

  ## calculate excitation function
  #calculate integral and approximate excitation function with gaussian distribution
  def get_gaus_f(self,m,l,tau,r): #takes int, tensor, int, float
      #trapezoid rule to integrate f(x)sin(mpix) from 0 to l
      #(f(a)+f(b))*(b-a)/2
      
      num_m = m
      m = tf.reshape(core.tf_float32(np.arange(1,m+1,1)),[1,m])
      x = self.approxnorm(l,l*r,0.1,tau) #l is a tensor here, x is a tensor too
      h = l/tau

      t = tf.reshape(core.tf_float32(tf.range(0,tau,1)),[1,tau]) #to replace the i
      # m by tau
      integral = (core.tf_float32(x[:-1])*tf.sin(tf.linalg.matrix_transpose(m)*np.pi*t*h/l)+core.tf_float32(x[1:])*tf.sin(tf.linalg.matrix_transpose(m)*np.pi*(t+1)*h/l))*h/2
      integral_sum = tf.reduce_sum(integral,axis=1)*2/l
   
      return tf.reshape(integral_sum,[1,num_m]) #return tensor

  ## calculate omega, sigma, K
  def getsigma(self,m1,m2,alpha,p,s11):  
      m1 = tf.reshape(core.tf_float32(tf.range(1,m1+1,1)),[1,m1])
      m2 = tf.reshape(core.tf_float32(tf.range(1,m2+1,1)),[1,m2])
      beta = alpha + 1/alpha

      sigma = s11 * (1 + p * ((tf.linalg.matrix_transpose(m1)*m1 * alpha + tf.linalg.matrix_transpose(m2)*m2/alpha) - beta))
      return sigma

  def getomega(self,m1,m2,alpha,p,D,w11,s11):
      m1 = tf.reshape(core.tf_float32(tf.range(1,m1+1,1)),[1,m1])
      m2 = tf.reshape(core.tf_float32(tf.range(1,m2+1,1)),[1,m2])

      interm = tf.linalg.matrix_transpose(m1)*m1 * alpha + tf.linalg.matrix_transpose(m2)*m2/alpha
      beta = alpha + 1/alpha

      omega_sq = D**2 * w11**2 * interm*interm + interm * (s11*s11 * (1 - p * beta)**2/beta + w11**2 * (1 - D**2 * beta**2)/beta) - s11*s11 * (1 - p * beta)**2
      omega = tf.math.sqrt(omega_sq)
      return tf.where(tf.math.is_nan(omega),tf.zeros_like(omega),omega) #check if there's zero in the omega matrix

  #k is dependent on striking position as well
  def get_gaus_k(self,m1,m2,omega,f1,f2, x1,x2,l,alpha): #take int, int, tensor, tensor, tensor, tensor, tensor, tensor, tensor
      #f1,f2 are tensors of (1,m1),(1,m2)
      m1 = tf.reshape(core.tf_float32(tf.range(1,m1+1,1)),[1,m1])
      m2 = tf.reshape(core.tf_float32(tf.range(1,m2+1,1)),[1,m2])

      l2 = l * alpha
      if float(x1) > float(l):
          x1 = 0
      if float(x2) > float(l2):
          x2 = 0
          
      k = (tf.linalg.matrix_transpose(f1) * f2) * (tf.linalg.matrix_transpose(tf.sin(m1 * np.pi * x1/l)) * tf.sin(m2 * np.pi * x2/l2))/ omega #assuming x1,x2 at center of the surface

      #k = np.round(k,10)
      return k #tensor of m by m

  ## calculate the ending sound 
  def getsounds_imp_gaus(self,m1,m2,r1,r2,w11,tau11,p,D,alpha,sr):
      """
      Calculate sound of a FTM 2D rectangular drum of shape {w,tau,p,D,alpha}
      excitation function:(Laplace) spatially gaussian, temporally delta
      modes: m1, m2 along each direction
      excitation position: r1, r2 
      """

      l = core.tf_float32(np.pi) #does this need to be tensor too?
      l2 = l*core.tf_float32(alpha)
      w11 = core.tf_float32(w11)
      s11 = -1/core.tf_float32(tau11)
      p = core.tf_float32(p)
      D = core.tf_float32(D)
      alpha = core.tf_float32(alpha)

      #position of striking
      x1 = l*r1 
      x2 = l2*r2
      #calculate intermediate variables sigma, omega, k 
      sigma= self.getsigma(m1,m2,alpha,p,s11)
      omega = self.getomega(m1,m2,alpha,p,D,w11,s11)
      k = self.get_gaus_k(m1,m2,omega,self.get_gaus_f(m1,1,300,r1),self.get_gaus_f(m2,alpha,300,r2),x1,x2,l,alpha)

      # define total number of samples
      dur = self.n_samples

      #sound can be roughly calculated as k * exp(sigma*t)*sin(omega*t)
      t = tf.reshape(core.tf_float32(tf.range(0,dur,1)),[1,1,dur])
      y = tf.reduce_sum(tf.reduce_sum(tf.reshape(k,[m1,m2,1]) * tf.math.exp(tf.reshape(sigma,[m1,m2,1]) * t/sr) * tf.sin(tf.reshape(omega,[m1,m2,1]) * t/sr),axis=0),axis=0)

      #normalize output sound
      y = y/max(y)


      return core.tf_float32(y)

  def get_controls(self,position_x,position_y,w_est,tau_est,p_est,D_est,alpha):
    """Convert network output tensors into a dictionary of synthesizer controls.

    Args:
      theta: 2-D tensor of shape parameters [batch, 5]
     

    Returns:
      controls: Dictionary of tensors of synthesizer controls.
    """
    alpha_est = 1 - alpha #correct the activation function 
    #transition position_x,position_y to the correct coordinates
    u1 = 0.5 * 0.5 * tf.minimum(position_x,position_y)
    u2 = 0.5 * 0.5 * tf.maximum(position_x,position_y)
    position_x = u1
    position_y = u2

    return {'position_x':position_x,
            'position_y':position_y,
            'w_est':w_est,
            'tau_est':tau_est,
            'p_est':p_est,
            'D_est':D_est,
            'alpha':alpha}

  def get_signal(self, mode, position_x,position_y,w_est,tau_est,p_est,D_est,alpha):
    """Synthesize audio with additive synthesizer from controls.

    Args:
      mode: number of modes
      position_x: float between 0 and 1, width proportion of the strike position
      position_y: float between 0 and 1, height proportion of the strike position
      w_est: fundamental frequency in hz
      tau_est: float 0.1~3 not in seconds
      p: float, roundness
      D: float, inharmonicity
      alpha: float between 0 and 1, ratio between width and height of the drum
      sample_rate: samples/second


    Returns:
      signal: A tensor of harmonic waves of shape [batch, n_samples].
    """
    signal = getsounds_imp_gaus(
      m1=self.mode,
      m2=self.mode,
      r1=position_x,
      r2=position_y,
      w11=w_est,
      tau11=tau_est,
      p=p_est,
      D=D_est,
      alpha=alpha,
      sr=self.sample_rate)
   
    return signal


