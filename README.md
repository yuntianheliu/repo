this code solve temperal diffusion equation with a artifical solution 
    u  =  exp(-t)*sin(pi*x)*sin(pi*y)
to run ,
mkdir build
cd build
cmake ../
mpirun -np 4 ./imp2d -n 100 -dt 0.0001
and use paraview to open the file solution%steps.vts
