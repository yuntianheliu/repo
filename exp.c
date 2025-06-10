#include <petscvec.h>
#include <math.h>

#define PI 3.14159265358979323846

PetscScalar exact_solution(PetscScalar x, PetscScalar t) {
    return exp(-t) * sin(PI * x);
}

PetscScalar source_term(PetscScalar x, PetscScalar t) {
    return (PI * PI - 1) * exp(-t) * sin(PI * x);
}

int main(int argc, char **argv) {
    PetscInitialize(&argc, &argv, NULL, NULL);

    PetscInt nx_global = 100;
    PetscScalar L = 1.0, D = 1.0, dx = L / (nx_global - 1);
    PetscScalar dt = 0.4 * dx * dx / D;
    PetscScalar T = 0.1;
    PetscInt nt = (PetscInt)(T / dt);

    MPI_Comm comm = PETSC_COMM_WORLD;
    PetscMPIInt rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Compute local size with overlap (ghost points)
    PetscInt local_nx = nx_global / size;
    PetscInt remainder = nx_global % size;
    if (rank < remainder) local_nx++;

    PetscInt start;
    MPI_Exscan(&local_nx, &start, 1, MPIU_INT, MPI_SUM, comm);
    if (rank == 0) start = 0;

    // Add 2 ghost points (left/right)
    PetscInt nx_with_ghosts = local_nx + 2;

    Vec u, u_new;
    VecCreateSeq(PETSC_COMM_SELF, nx_with_ghosts, &u);
    VecDuplicate(u, &u_new);

    PetscScalar *u_array, *u_new_array;
    VecGetArray(u, &u_array);
    VecGetArray(u_new, &u_new_array);

    // Set initial condition (local only, ignoring ghosts)
    for (PetscInt i = 1; i <= local_nx; i++) {
        PetscInt gi = start + i - 1;          // global index
        PetscScalar x = gi * dx;
        u_array[i] = sin(PI * x);
    }

    u_array[0] = 0.0;                        // Left ghost
    u_array[local_nx + 1] = 0.0;             // Right ghost

    // Time loop
    for (PetscInt step = 0; step < nt; step++) {
        PetscScalar t = step * dt;

        // Communicate ghost points
        MPI_Request reqs[4];
        if (rank > 0) {
            MPI_Isend(&u_array[1], 1, MPIU_SCALAR, rank - 1, 0, comm, &reqs[0]);
            MPI_Irecv(&u_array[0], 1, MPIU_SCALAR, rank - 1, 1, comm, &reqs[1]);
        }
        if (rank < size - 1) {
            MPI_Isend(&u_array[local_nx], 1, MPIU_SCALAR, rank + 1, 1, comm, &reqs[2]);
            MPI_Irecv(&u_array[local_nx + 1], 1, MPIU_SCALAR, rank + 1, 0, comm, &reqs[3]);
        }

        if (rank > 0) MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        if (rank < size - 1) MPI_Waitall(2, reqs + 2, MPI_STATUSES_IGNORE);

        // Update interior points
        for (PetscInt i = 1; i <= local_nx; i++) {
            PetscInt gi = start + i - 1;
            PetscScalar x = gi * dx;
            PetscScalar lap = (u_array[i - 1] - 2 * u_array[i] + u_array[i + 1]) / (dx * dx);
            PetscScalar s = source_term(x, t);
            u_new_array[i] = u_array[i] + dt * (D * lap + s);
        }

        // Apply BCs
        if (rank == 0) u_new_array[1] = 0.0;
        if (rank == size - 1) u_new_array[local_nx] = 0.0;

        // Swap arrays
        PetscScalar *tmp = u_array;
        u_array = u_new_array;
        u_new_array = tmp;
    }

    VecRestoreArray(u, &u_array);
    VecRestoreArray(u_new, &u_new_array);

    // Compute L1 error locally
    PetscScalar local_err = 0.0;
    VecGetArray(u, &u_array);
    for (PetscInt i = 1; i <= local_nx; i++) {
        PetscInt gi = start + i - 1;
        PetscScalar x = gi * dx;
        PetscScalar err = fabs(u_array[i] - exact_solution(x, T));
        local_err += err;
    }
    VecRestoreArray(u, &u_array);

    // Global error
    PetscScalar global_err = 0.0;
    MPI_Allreduce(&local_err, &global_err, 1, MPIU_SCALAR, MPI_SUM, comm);

    if (rank == 0)
        PetscPrintf(comm, "explicit euler L1 error at t=%.2f: %.6e\n", T, global_err / nx_global);

    VecDestroy(&u);
    VecDestroy(&u_new);
    PetscFinalize();
    return 0;
}

