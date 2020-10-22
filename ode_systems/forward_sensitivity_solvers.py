import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import torch
from torch.autograd import functional

class ForwardSensManualJacobians(object):
    def __init__(self, rhs, jac_x, jac_p, n_states, n_params, times,
                 rtol=1e-5, atol=1e-6, y0=None):
        super(ForwardSensManualJacobians, self).__init__()

        if y0 is None:
            self._y0 = np.array(np.zeros(n_states), dtype=float)
        else:
            self._y0 = np.array(y0, dtype=float) 
        self._rhs = rhs
        self._jac_x = jac_x
        self._jac_p = jac_p
        
        self._n_states = n_states
        self._n_params = n_params
        self._times = times
        self._T = len(times)
        self._rtol = rtol
        self._atol = atol  
        self._unknown_y0 = False
        self._dy0dy0 = None

    def set_y0(self, y0):
      self._y0 = y0
    
    def set_unknown_y0(self):
        self._unknown_y0 = True
        self._dy0dy0 = [i for i in range(self._n_states)]
        for s in range(self._n_states):
          self._dy0dy0.append((s+1)*(self._n_states + (self._n_params-self._n_states)) + s)
    
    def _rhs_ivp(self, t, y, p):
      return self._rhs(y, t, p)
    
    def solve(self, parameters):
        return self._solve(parameters, False)

    def solve_with_sensitivities(self, parameters):
        return self._solve(parameters, True)

    def _solve(self, parameters, sensitivities):
        
        if sensitivities:
            def augmented_ode(t, y_and_dydp, p):                
                y = y_and_dydp[0:self._n_states]
                dydp = y_and_dydp[self._n_states:].reshape((
                    self._n_states, self._n_params))
                
                dydt = self._rhs(y, t, p)
                d_dydp_dt = np.matmul(self._jac_x(y, p), dydp) + self._jac_p(y, p)
                return np.concatenate((dydt, d_dydp_dt.reshape(-1)))    
            
            y0 = np.zeros(self._n_states + (self._n_states * self._n_params))
            
            if self._unknown_y0:
                y0[self._dy0dy0[self._n_states:]] = 1.
                y0[self._dy0dy0[:self._n_states]] = parameters[-self._n_states:]
            else:
                y0[0:self._n_states] = self._y0
            
            sol = solve_ivp(augmented_ode, [self._times[0], self._times[-1]], \
                            y0, t_eval = self._times, method = 'LSODA', args=(parameters,), \
                            rtol=self._rtol, atol=self._atol)
            
            result = sol.y.T
            x = result[:, 0:self._n_states]
            dx_dp = result[:, self._n_states:].reshape((
                self._T, self._n_states, self._n_params))
            return x, dx_dp
        else:
            return solve_ivp(self._rhs_ivp, [self._times[0], self._times[-1]], \
                            self._y0, t_eval = self._times, method = 'LSODA', args=(parameters,), \
                            rtol=self._rtol, atol=self._atol).y.T
                            
class ForwardSensTorchJacobians(object):
    def __init__(self, rhs, n_states, n_params, times,
                 rtol=1e-5, atol=1e-6, y0=None):
        super(ForwardSensTorchJacobians, self).__init__()

        if y0 is None:
            self._y0 = np.array(np.zeros(n_states), dtype=float)
        else:
            self._y0 = np.array(y0, dtype=float) 
        self._rhs = rhs
        
        self._n_states = n_states
        self._n_params = n_params
        self._times = times
        self._T = len(times)
        self._rtol = rtol
        self._atol = atol  
        self._unknown_y0 = False
        self._dy0dy0 = None

    def set_y0(self, y0):
      self._y0 = y0
    
    def set_unknown_y0(self):
        self._unknown_y0 = True
        self._dy0dy0 = [i for i in range(self._n_states)]
        for s in range(self._n_states):
          self._dy0dy0.append((s+1)*(self._n_states + (self._n_params-self._n_states)) + s)
    
    def _rhs_ivp(self, t, y, p):
      return self._rhs(y, t, p)
    
    def solve(self, parameters):
        return self._solve(parameters, False)

    def solve_with_sensitivities(self, parameters):
        return self._solve(parameters, True)

    def _solve(self, parameters, sensitivities):
        
        if sensitivities:
            def augmented_ode(t, y_and_dydp, p):                
                y = y_and_dydp[0:self._n_states]
                dydp = y_and_dydp[self._n_states:].reshape((
                    self._n_states, self._n_params))
                with torch.enable_grad():
                    t_ = torch.as_tensor(t, dtype=torch.float)
                    y_ = torch.as_tensor(y, dtype=torch.float)
                    p_ = torch.as_tensor(p, dtype=torch.float)
                    jac_x, _, jac_p = functional.jacobian(lambda y,t,p,tch=True: self._rhs(y,t,p,tch),(y_,t_,p_))
                d_dydp_dt = np.matmul(jac_x.detach().numpy(), dydp) + jac_p.detach().numpy()
                dydt = self._rhs(y, t, p)                
                return np.concatenate((dydt, d_dydp_dt.reshape(-1)))
            
            y0 = np.zeros(self._n_states + (self._n_states * self._n_params))
            
            if self._unknown_y0:
                y0[self._dy0dy0[self._n_states:]] = 1.
                y0[self._dy0dy0[:self._n_states]] = parameters[-self._n_states:]
            else:
                y0[0:self._n_states] = self._y0
            
            sol = solve_ivp(augmented_ode, [self._times[0], self._times[-1]], \
                            y0, t_eval = self._times, method = 'LSODA', args=(parameters,), \
                            rtol=self._rtol, atol=self._atol)
            
            result = sol.y.T
            x = result[:, 0:self._n_states]
            dx_dp = result[:, self._n_states:].reshape((
                self._T, self._n_states, self._n_params))
            return x, dx_dp
        else:
            return solve_ivp(self._rhs_ivp, [self._times[0], self._times[-1]], \
                            self._y0, t_eval = self._times, method = 'LSODA', args=(parameters,), \
                            rtol=self._rtol, atol=self._atol).y.T

                          