#include <iostream>
#include <vector>
#include <stdio.h>      
#include <math.h>       
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
namespace py = pybind11;
/*-----------------------------------------------------------
Function for SSA (Gillespie algorithm), 
for the stochastic Lotka-Volterra model 
 ------------------------------------------------------------- */
 std::array<int, 2>  LV(py::array_t<double> rates, 
   py::array_t<int> init_molecules, double_t tsart, double_t tend){
  // set fixed elements:
  auto c = rates.unchecked<1>();
  auto x0 = init_molecules.unchecked<1>();
  std::array<int, 2> x ={x0(0), x0(1)};
  double h0,h1,h2,h3,u1,u2,dt;
  double t = tsart;
  std::random_device rd;  
  std::mt19937 gen(rd()); 

  std::uniform_real_distribution<double> u(0.0, 1.0);

  while (1==1)
  {
    h1=x[0]*c(0); h2=c(1)*x[0]*x[1]; h3=c(2)*x[1];
    h0=h1+h2+h3;
 
    
    if ((h0<1e-10)||(x[0]>=1000000))
      t=1e99;
    else{
      u1 = u(gen);
      dt = -log(u1) / h0;
      t += dt;
    }
    if (t>=tend) {      
      return x;
    }
    u2 = u(gen);
    if (u2<h1/h0)
      x[0]+=1;
    else if (u2<(h1+h2)/h0) {
      x[0]-=1; x[1]+=1;
    } else
      x[1]-=1;
  }
}
PYBIND11_PLUGIN(lvssa) {
    pybind11::module m("lvssa", "auto-compiled c++ extension");
    m.def("LV", &LV, py::return_value_policy::reference_internal);
    return m.ptr();
}
