this code solve temperal diffusion equation with a artifical solution 
                 u  =  exp(-t)*sin(pi*x)*sin(pi*y)

to compile,
mkdir build
cd build
cmake ../
make

to run,
mpirun -n 4 ./imp2d -t0 0.0 -T 1.0 -N 100 -dt 0.01 -output_interval 20 -restart_interval 50 -restart -1
t0 means start from time, must be 0 for the first run
T means end time,
restart 1 means restart from t0, then run
mpirun -n 4 ./imp2d -t0 0.5 -T 1.5 -N 100 -dt 0.01 -output_interval 20 -restart_interval 50 -restart 1
the latest time u field is dumped into diffusion2d.h5 file, if add '-restart 1' in the command, it can read the latest time u field and restart.

to view,
paraview
open file 'solution%steps.vts'

