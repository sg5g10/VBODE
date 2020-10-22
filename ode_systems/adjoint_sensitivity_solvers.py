import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import torch
from torch.autograd import functional

class AdjointSensManualJacobians(object):
  def __init__(self, _rhs_f, _jac_x_f, _jac_p_f, n_states, n_params, times,
                rtol=1e-5, atol=1e-5, y0=None):  
    super(AdjointSensManualJacobians, self).__init__()

    if y0 is None:
        self._y0 = np.array(np.zeros(n_states), dtype=float)
    else:
        self._y0 = np.array(y0, dtype=float) 
    self._rhs = _rhs_f
    self._jac_x = _jac_x_f
    self._jac_p = _jac_p_f
    
    self._n_states = n_states
    self._n_params = n_params
    self._times = times
    self._T = len(times)
    self._rtol = rtol
    self._atol = atol  
    self.nfev = 0
    self._unknown_y0 = False
    self._checkpointed = False

  def set_y0(self, y0):
    self._y0 = y0
  
  def set_unknown_y0(self):
    self._unknown_y0 = True

  def set_checkpointed(self):
    self._checkpointed = True

  def _rhs_ivp(self, t, y, p):
      return self._rhs(y, t, p)

  def _jac_x_ivp(self, t, y, p):
    return self._jac_x(y, p)
          
  def solve(self, parameters):
    return self._solve(parameters, None, None, None, False)

  def solve_with_sensitivities(self, parameters, yt, g, sol):
    return self._solve(parameters, yt, g, sol, True)

  def _solve(self, parameters, yt, g, sol, sensitivities):
    if sensitivities:
      vjp_params = np.zeros(self._n_params)

      def augmented_dynamics(t, aug_state, p, interpolation):            
        adj = aug_state[:self._n_states]
        y = interpolation.sol(t)    
                   
        jac_p = self._jac_p(y, p)
        jac_x = self._jac_x(y, p)
        dL_y = np.dot(-adj, jac_x)
        dL_p = np.dot(-adj, jac_p)  
        return np.hstack((dL_y, dL_p))
      
      vjp_y = g[-1, :]         
      
      for i in range(self._T - 1, 0, -1):
        if self._checkpointed:
          checkpoint_interpolation = solve_ivp(self._rhs_ivp, \
                                             [self._times[i-1], self._times[i]], yt[i-1,:], \
                                             method = 'LSODA', args=(parameters,), \
                                             dense_output=True, rtol=self._rtol, atol=self._atol)
        else:
          checkpoint_interpolation = sol                                           
        aug_y0 = np.hstack((vjp_y, vjp_params))
        aug_ans = solve_ivp(augmented_dynamics, [self._times[i], self._times[i-1]], \
                            aug_y0, method = 'LSODA', args=(parameters,checkpoint_interpolation), \
                            rtol=self._rtol, atol=self._atol)
        
        self.nfev += aug_ans.nfev
        
        vjp_y, vjp_params = aug_ans.y[:self._n_states,-1], aug_ans.y[self._n_states:,-1]
        vjp_y = vjp_y + g[i-1,:]
      
      return np.hstack([vjp_params[:(self._n_params-self._n_states)], vjp_y]) if self._unknown_y0 else vjp_params            
    else:
      if self._unknown_y0:
        y0 = parameters[-self._n_states:]
      else:
        y0 = self._y0
      return solve_ivp(self._rhs_ivp, [self._times[0], self._times[-1]], y0, \
                       t_eval = self._times, method = 'LSODA', args=(parameters,), \
                       rtol=self._rtol, atol=self._atol).y.T if self._checkpointed else solve_ivp(self._rhs_ivp, 
                      [self._times[0], self._times[-1]], y0, \
                       t_eval = self._times, method = 'LSODA', args=(parameters,), \
                       dense_output=True, rtol=self._rtol, atol=self._atol)


class AdjointSensTorchJacobians(object):
  def __init__(self, _rhs_f, n_states, n_params, times,
                rtol=1e-5, atol=1e-5, y0=None):  
    super(AdjointSensTorchJacobians, self).__init__()

    if y0 is None:
        self._y0 = np.array(np.zeros(n_states), dtype=float)
    else:
        self._y0 = np.array(y0, dtype=float) 
    self._rhs = _rhs_f
    
    self._n_states = n_states
    self._n_params = n_params
    self._times = times
    self._T = len(times)
    self._rtol = rtol
    self._atol = atol  
    self.nfev = 0
    self._unknown_y0 = False
    self._checkpointed = False

  def set_y0(self, y0):
    self._y0 = y0
  
  def set_unknown_y0(self):
    self._unknown_y0 = True

  def set_checkpointed(self):
    self._checkpointed = True

  def _rhs_ivp(self, t, y, p):
      return self._rhs(y, t, p)
        
  def solve(self, parameters):
    return self._solve(parameters, None, None, None, False)

  def solve_with_sensitivities(self, parameters, yt, g, sol):
    return self._solve(parameters, yt, g, sol, True)

  def _solve(self, parameters, yt, g, sol, sensitivities):
    if sensitivities:
      vjp_params = np.zeros(self._n_params)

      def augmented_dynamics(t, aug_state, p, interpolation):            
        adj = aug_state[:self._n_states]
        y = interpolation.sol(t)    
        with torch.enable_grad():
          t_ = torch.as_tensor(t, dtype=torch.float)
          y_ = torch.as_tensor(y, dtype=torch.float)
          p_ = torch.as_tensor(p, dtype=torch.float)
          adj_ = torch.as_tensor(adj, dtype=torch.float)          
          dL_y,_,dL_p = functional.vjp(lambda y,t,p,tch=True: self._rhs(y,t,p,tch),(y_,t_,p_), -adj_, 
                                strict=False, create_graph=False)[1]
        return np.hstack((dL_y.detach().numpy(), dL_p.detach().numpy()))
      
      vjp_y = g[-1, :]         
      
      for i in range(self._T - 1, 0, -1):
        if self._checkpointed:
          checkpoint_interpolation = solve_ivp(self._rhs_ivp, \
                                             [self._times[i-1], self._times[i]], yt[i-1,:], \
                                             method = 'LSODA', args=(parameters,), \
                                             dense_output=True, rtol=self._rtol, atol=self._atol)
        else:
          checkpoint_interpolation = sol                                           
        aug_y0 = np.hstack((vjp_y, vjp_params))
        aug_ans = solve_ivp(augmented_dynamics, [self._times[i], self._times[i-1]], \
                            aug_y0, method = 'LSODA', args=(parameters,checkpoint_interpolation), \
                            rtol=self._rtol, atol=self._atol)
        
        self.nfev += aug_ans.nfev
        
        vjp_y, vjp_params = aug_ans.y[:self._n_states,-1], aug_ans.y[self._n_states:,-1]
        vjp_y = vjp_y + g[i-1,:]
      
      return np.hstack([vjp_params[:(self._n_params-self._n_states)], vjp_y]) if self._unknown_y0 else vjp_params            
    else:
      if self._unknown_y0:
        y0 = parameters[-self._n_states:]
      else:
        y0 = self._y0
      return solve_ivp(self._rhs_ivp, [self._times[0], self._times[-1]], y0, \
                       t_eval = self._times, method = 'LSODA', args=(parameters,), \
                       rtol=self._rtol, atol=self._atol).y.T if self._checkpointed else solve_ivp(self._rhs_ivp, 
                      [self._times[0], self._times[-1]], y0, \
                       t_eval = self._times, method = 'LSODA', args=(parameters,), \
                       dense_output=True, rtol=self._rtol, atol=self._atol)     

                          