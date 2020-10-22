import numpy as np
from joblib import Parallel, delayed
import torch
from torch.autograd import Function
import time as timer

def sens(i, model, param):
  return model.solve_with_sensitivities(param[i,:])

class ForwardSensOp(Function):
  @staticmethod
  def forward(ctx, params, args): 
    if len(args)>1:
      ode_model, init_vals = args
      ode_model.set_y0(init_vals.detach().numpy())
    else:
      ode_model = args[0]      
      
    numpy_params = params.detach().numpy()
    
    ctx.T, ctx.k, ctx.d = ode_model._T, ode_model._n_states, ode_model._n_params 
    if numpy_params.ndim != 2:
      raise ValueError('Parameters should be batch size (L) x dimension (D)')
    else:          
      VECTORISE = False
      if numpy_params.shape[0] == 1:   
        ctx.V = VECTORISE
        sol, Jac = ode_model.solve_with_sensitivities(numpy_params.reshape((numpy_params.shape[1],)))       
        ctx.save_for_backward(torch.as_tensor(Jac, dtype=params.dtype))

      elif numpy_params.shape[0] > 1:                
        VECTORISE = True
        ctx.V = VECTORISE
        ctx.L = numpy_params.shape[0]
        
      
        sols = Parallel(n_jobs=numpy_params.shape[0])(delayed(
              ode_model.solve_with_sensitivities)(numpy_params[i,:]) 
                                  for i in range(numpy_params.shape[0]))


        sol = np.stack([sols[i][0] for i in range(numpy_params.shape[0])], axis=0)
        Jac = np.stack([sols[i][1] for i in range(numpy_params.shape[0])], axis=0)  
                 
        ctx.save_for_backward(torch.as_tensor(Jac, dtype=params.dtype))
      return torch.as_tensor(sol, dtype=params.dtype)        

  @staticmethod
  def backward(ctx, grad_output):
    T = ctx.T
    d = ctx.d
    k = ctx.k
    VECTORISE = ctx.V
    bJac = ctx.saved_tensors[0]
    if not(VECTORISE):        
      numpy_Jac = bJac.detach().numpy().reshape((k*T,d))
      g = grad_output.detach().numpy().reshape(k*T)
      vjp = g.T.dot(numpy_Jac).reshape((-1,d))
    else:       
      L = ctx.L
      numpy_Jac = bJac.detach().numpy().reshape((L,k*T,d))
      g = grad_output.detach().numpy().reshape(L,k*T)
      vjp = np.array([g[i].T.dot(numpy_Jac[i]) for i in range(L)]).reshape((L,d))           
    return grad_output.new(torch.as_tensor(vjp)), None#torch.from_numpy(vjp), None

class AdjointSensOp(Function):  
  @staticmethod
  def forward(ctx, params, args): 
    if len(args)>1:
      ode_model, init_vals = args
      ode_model.set_y0(init_vals.detach().numpy())
    else:
      ode_model = args[0]         
    numpy_params = params.detach().numpy()
    ctx.ode_model, ctx.T, ctx.k, ctx.d = ode_model, ode_model._T, ode_model._n_states, ode_model._n_params 
    if numpy_params.ndim != 2:
      raise ValueError('Parameters should be batch size (L) x dimension (D)')
    else:          
      VECTORISE = False
      if numpy_params.shape[0] == 1:   
        ctx.V = VECTORISE
        with np.errstate(all='ignore'):
          sol = ode_model.solve(numpy_params.reshape((numpy_params.shape[1],))) 
        if not(ode_model._checkpointed):
          states = sol.sol(ode_model._times).T  
          ctx.sol = sol  
        else:
          states = sol

      elif numpy_params.shape[0] > 1:                
        VECTORISE = True
        ctx.V = VECTORISE
        ctx.L = numpy_params.shape[0]
        
        with np.errstate(all='ignore'):
          
          sols = Parallel(n_jobs=numpy_params.shape[0])(delayed(
                ode_model.solve)(numpy_params[i,:]) 
                                    for i in range(numpy_params.shape[0]))

        if not(ode_model._checkpointed):              
          ctx.sol = sols
          states = np.stack([sols[i].sol(ode_model._times).T for i in range(numpy_params.shape[0])], axis=0)  
        else:
          states = np.stack([sols[i] for i in range(numpy_params.shape[0])], axis=0)

    ctx.save_for_backward(params, torch.as_tensor(states, dtype=params.dtype))
    return torch.as_tensor(states, dtype=params.dtype)        

  @staticmethod
  def backward(ctx, grad_output):
    ode_model = ctx.ode_model
    T = ctx.T
    d = ctx.d
    k = ctx.k
    if not(ode_model._checkpointed):
      sol = ctx.sol
    VECTORISE = ctx.V
    
    params = ctx.saved_tensors[0] 
    states = ctx.saved_tensors[1]

    if not(VECTORISE): 
      g = grad_output.detach().numpy()  
      numpy_p = params.detach().numpy()
      numpy_p = numpy_p.reshape((numpy_p.shape[1],))
      numpy_states = states.detach().numpy()  
      with np.errstate(all='ignore'):
        if not(ode_model._checkpointed):
          vjp = ode_model.solve_with_sensitivities(numpy_p, numpy_states, g, sol).reshape((-1,d))
        else:
          vjp = ode_model.solve_with_sensitivities(numpy_p, numpy_states, g, None).reshape((-1,d))

    else:       
      L = ctx.L 
      numpy_p = params.detach().numpy().reshape((L,d))
      numpy_states = states.detach().numpy().reshape((L,T,k))
      g = grad_output.detach().numpy() 
      with np.errstate(all='ignore'):
        if not(ode_model._checkpointed):
          vjps = Parallel(n_jobs=L)(delayed(
              ode_model.solve_with_sensitivities)(numpy_p[i,:], \
                                        numpy_states[i,:,:], g[i,:,:], sol[i]) 
                                      for i in range(L))
        else:
          vjps = Parallel(n_jobs=L)(delayed(
              ode_model.solve_with_sensitivities)(numpy_p[i,:], \
                                        numpy_states[i,:,:], g[i,:,:], None) 
                                      for i in range(L))
      
      vjp = np.array(vjps).reshape((L,d))
    return grad_output.new(torch.as_tensor(vjp, dtype=params.dtype)), None