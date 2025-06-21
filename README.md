this code solve temperal diffusion equation with a artifical solution 
                 u  =  exp(-t)*sin(pi*x)*sin(pi*y)

to compile,
mkdir build
cd build
cmake ../
make

to run,
mpirun -np 4 ./imp2d -n 100 -dt 0.0001
the latest time u field is dumped into diffusion2d.h5 file, if add '-restart 1' in the command, it can read the latest time u field and restart.

to view,
paraview
open file 'solution%steps.vts'

