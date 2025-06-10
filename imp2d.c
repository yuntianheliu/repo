#include <petscksp.h>
#include <math.h>

// Exact solution and source
PetscScalar u_exact(PetscReal x, PetscReal y, PetscReal t) {
    return PetscExpReal(-t) * PetscSinReal(PETSC_PI * x) * PetscSinReal(PETSC_PI * y);
}

PetscScalar source_term(PetscReal x, PetscReal y, PetscReal t, PetscReal D) {
    return (-1.0 + 2.0 * PETSC_PI * PETSC_PI * D) * u_exact(x, y, t);
}

void WriteVTK2D(const char *filename, Vec u, PetscInt nx, PetscInt ny, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    PetscScalar *local_data;
    PetscInt Istart, Iend, n_local;
    VecGetOwnershipRange(u, &Istart, &Iend);
    VecGetLocalSize(u, &n_local);
    VecGetArray(u, &local_data);

    PetscScalar *global_data = NULL;
    PetscInt *recvcounts = NULL, *displs = NULL;
    if (rank == 0) {
        global_data = (PetscScalar *)malloc(nx * ny * sizeof(PetscScalar));
        recvcounts = (PetscInt *)malloc(size * sizeof(PetscInt));
        displs = (PetscInt *)malloc(size * sizeof(PetscInt));
    }

    // Gather sizes
    PetscInt mycount = n_local;
    MPI_Gather(&mycount, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, comm);


    // Gather displacements
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i)
            displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    // Gather full solution on rank 0
    MPI_Gatherv(local_data, n_local, MPIU_SCALAR,
                global_data, recvcounts, displs, MPIU_SCALAR, 0, comm);

    VecRestoreArray(u, &local_data);

    // Rank 0 writes VTK
    if (rank == 0) {
        FILE *f = fopen(filename, "w");
        fprintf(f, "<?xml version=\"1.0\"?>\n");
        fprintf(f, "<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
        fprintf(f, "  <StructuredGrid WholeExtent=\"0 %d 0 %d 0 0\">\n", nx - 1, ny - 1);
        fprintf(f, "    <Piece Extent=\"0 %d 0 %d 0 0\">\n", nx - 1, ny - 1);
        fprintf(f, "      <Points>\n");
        fprintf(f, "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n");
        for (PetscInt j = 0; j < ny; j++) {
            for (PetscInt i = 0; i < nx; i++) {
                PetscReal x = (i + 1.0) / (nx + 1);
                PetscReal y = (j + 1.0) / (ny + 1);
                fprintf(f, "          %.6f %.6f 0.0\n", x, y);
            }
        }
        fprintf(f, "        </DataArray>\n");
        fprintf(f, "      </Points>\n");

        fprintf(f, "      <PointData Scalars=\"u\">\n");
        fprintf(f, "        <DataArray type=\"Float32\" Name=\"u\" format=\"ascii\">\n");
        for (PetscInt i = 0; i < nx * ny; i++) {
            fprintf(f, "          %.6f\n", PetscRealPart(global_data[i]));
        }
        fprintf(f, "        </DataArray>\n");
        fprintf(f, "      </PointData>\n");

        fprintf(f, "    </Piece>\n");
        fprintf(f, "  </StructuredGrid>\n");
        fprintf(f, "</VTKFile>\n");
        fclose(f);

        free(global_data);
        free(recvcounts);
        free(displs);
    }
}

void ComputeL2Error(Vec u, PetscInt nx, PetscInt ny, PetscReal time, MPI_Comm comm) {
    PetscInt Istart, Iend;
    VecGetOwnershipRange(u, &Istart, &Iend);

    PetscScalar *u_array;
    VecGetArray(u, &u_array);

    PetscReal dx = 1.0 / (nx + 1), dy = 1.0 / (ny + 1);
    PetscReal local_sum = 0.0;

    for (PetscInt Ii = Istart; Ii < Iend; ++Ii) {
        PetscInt j = Ii / nx;
        PetscInt i = Ii % nx;

        PetscReal x = (i + 1) * dx;
        PetscReal y = (j + 1) * dy;

        PetscReal u_exact_val = exp(-time) * sin(PETSC_PI * x) * sin(PETSC_PI * y);
        PetscReal diff = PetscRealPart(u_array[Ii - Istart]) - u_exact_val;
        local_sum += diff * diff;
        // PetscPrintf(comm, " %.3f\n", diff); // Optional: comment out to reduce output
    }

    VecRestoreArray(u, &u_array);

    PetscReal global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPIU_REAL, MPI_SUM, comm);

    PetscReal L2 = sqrt(global_sum * dx * dy);

    int rank;
    MPI_Comm_rank(comm, &rank);
    PetscPrintf(comm, "Rank %d | t = %.3f | L2 error = %.6e\n", rank, time, L2);
}


int main(int argc, char **args) {
    PetscInitialize(&argc, &args, NULL, NULL);

    PetscInt nx = 10, ny = 10;
    PetscReal dt = 0.0001, T = 1.0;
    PetscReal D = 1.0;  // diffusion coefficient
    PetscReal dx, dy;
    PetscInt steps = (PetscInt)(T / dt);
    PetscInt outputstep = steps/5;

    MPI_Comm comm = PETSC_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Grid spacing
    dx = 1.0 / (nx + 1);
    dy = 1.0 / (ny + 1);
    PetscInt N = nx * ny;

    // Create solution vector
    Vec u, b, u_ex;
    VecCreate(comm, &u);
    VecSetSizes(u, PETSC_DECIDE, N);
    VecSetFromOptions(u);
    VecDuplicate(u, &b);
    VecDuplicate(u, &u_ex);

    // Create matrix
    Mat A;
    MatCreate(comm, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N);
    MatSetFromOptions(A);
    MatMPIAIJSetPreallocation(A, 5, NULL, 5, NULL);
    MatSeqAIJSetPreallocation(A, 5, NULL);
    MatSetUp(A);

    // Ownership range
    PetscInt Istart, Iend;
    VecGetOwnershipRange(u, &Istart, &Iend);

    // Fill matrix A (backward Euler)
    for (PetscInt Ii = Istart; Ii < Iend; ++Ii) {
        PetscInt j = Ii / nx;
        PetscInt i = Ii % nx;

        PetscInt row = Ii;
        PetscReal diag = 1.0 + dt * D * (2.0 / (dx * dx) + 2.0 / (dy * dy));
        PetscScalar v[5];
        PetscInt col[5];
        PetscInt ncols = 0;

        // Center
        v[ncols] = diag; col[ncols] = row; ncols++;
        // West
        if (i > 0) { v[ncols] = -dt * D / (dx * dx); col[ncols] = row - 1; ncols++; }
        // East
        if (i < nx - 1) { v[ncols] = -dt * D / (dx * dx); col[ncols] = row + 1; ncols++; }
        // South
        if (j > 0) { v[ncols] = -dt * D / (dy * dy); col[ncols] = row - nx; ncols++; }
        // North
        if (j < ny - 1) { v[ncols] = -dt * D / (dy * dy); col[ncols] = row + nx; ncols++; }

        MatSetValues(A, 1, &row, ncols, col, v, INSERT_VALUES);
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    // Initial condition
    PetscScalar *u_array;
    VecGetArray(u, &u_array);
    for (PetscInt Ii = Istart; Ii < Iend; ++Ii) {
        PetscInt j = Ii / nx;
        PetscInt i = Ii % nx;
        PetscReal x = (i + 1) * dx;
        PetscReal y = (j + 1) * dy;
        u_array[Ii - Istart] = u_exact(x, y, 0.0);
    }
    VecRestoreArray(u, &u_array);

    // Solver
    KSP ksp;
    KSPCreate(comm, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetFromOptions(ksp);

    // Time stepping
    for (PetscInt step = 1; step <= steps; ++step) {
        PetscReal t = step * dt;

        // Assemble RHS
        VecGetArray(u, &u_array);
        PetscScalar *b_array;
        VecGetArray(b, &b_array);

        for (PetscInt Ii = Istart; Ii < Iend; ++Ii) {
            PetscInt j = Ii / nx;
            PetscInt i = Ii % nx;
            PetscReal x = (i + 1) * dx;
            PetscReal y = (j + 1) * dy;
            b_array[Ii - Istart] = u_array[Ii - Istart] + dt * source_term(x, y, t, D);
        }

        VecRestoreArray(b, &b_array);
        VecRestoreArray(u, &u_array);

        // Solve A u^{n+1} = b
        KSPSolve(ksp, b, u);
        
        if(step % outputstep == 0) {
            char filename[256];
            snprintf(filename, sizeof(filename), "solution_t%03d.vts", step);
            WriteVTK2D(filename, u, nx, ny, PETSC_COMM_WORLD);
            
            ComputeL2Error(u, nx, ny, t, PETSC_COMM_WORLD);
            }
        
    }
    
    

/*
    // Output: gather and write from rank 0
    PetscScalar *u_local;
    PetscInt local_size;
    VecGetLocalSize(u, &local_size);
    VecGetArray(u, &u_local);

    PetscScalar *u_global = NULL;
    PetscInt *recvcounts = NULL, *displs = NULL;

    if (rank == 0) {
        u_global = (PetscScalar *)malloc(N * sizeof(PetscScalar));
        recvcounts = (PetscInt *)malloc(size * sizeof(PetscInt));
        displs = (PetscInt *)malloc(size * sizeof(PetscInt));
    }

    MPI_Gather(&local_size, 1, MPIU_INT, recvcounts, 1, MPIU_INT, 0, comm);
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i)
            displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    MPI_Gatherv(u_local, local_size, MPIU_SCALAR,
                u_global, recvcounts, displs, MPIU_SCALAR,
                0, comm);
    VecRestoreArray(u, &u_local);

    if (rank == 0) {
        FILE *f = fopen("solution2d.dat", "w");
        for (PetscInt j = 0; j < ny; ++j) {
            for (PetscInt i = 0; i < nx; ++i) {
                PetscInt idx = j * nx + i;
                PetscReal x = (i + 1) * dx;
                PetscReal y = (j + 1) * dy;
                fprintf(f, "%g %g %g\n", x, y, PetscRealPart(u_global[idx]));
            }
        }
        fclose(f);
        free(u_global);
        free(recvcounts);
        free(displs);
    }
    
    */
    


    // Clean up
    KSPDestroy(&ksp);
    VecDestroy(&u);
    VecDestroy(&b);
    MatDestroy(&A);
    PetscFinalize();
    return 0;
}

