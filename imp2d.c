#include <petscksp.h>
#include <math.h>
#include <hdf5.h>
#include <petscsys.h>
#include <petscviewerhdf5.h>

#define FILE_NAME "diffusion2d.h5"
#define DATASET_NAME "solution"

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
    }

    VecRestoreArray(u, &u_array);

    PetscReal global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPIU_REAL, MPI_SUM, comm);

    PetscReal L2 = sqrt(global_sum * dx * dy);

    int rank;
    MPI_Comm_rank(comm, &rank);
    PetscPrintf(comm, "Rank %d | t = %.3f | L2 error = %.6e\n", rank, time, L2);
}

void writeh5(Vec u, PetscInt nx, PetscInt ny, PetscInt step) {
    MPI_Comm comm = PETSC_COMM_WORLD;
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

    // Gather full vector on rank 0
    MPI_Gatherv(local_data, n_local, MPIU_SCALAR,
                global_data, recvcounts, displs, MPIU_SCALAR, 0, comm);

    VecRestoreArray(u, &local_data);

    if (rank == 0) {
        // Write to HDF5
        hid_t file_id, dataset_id, dataspace_id;
        hsize_t dims[2] = {nx, ny};

        char filename[256];
	snprintf(filename, sizeof(filename), "restart_t%03d.h5", step);
	file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

        dataspace_id = H5Screate_simple(2, dims, NULL);
        dataset_id = H5Dcreate(file_id, DATASET_NAME, H5T_NATIVE_DOUBLE,
                               dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        // NOTE: global_data is in row-major layout, i.e., index = j*nx + i
        // But Vec index = j*nx + i, so it's compatible
        H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                 H5P_DEFAULT, global_data);
                 
        // Save the current time step as attributeqqqqqqqqqqqqqqqq
        // Create and write attribute on the dataset, not the file
        hid_t attr_space = H5Screate(H5S_SCALAR);
        hid_t attr_id = H5Acreate(dataset_id, "step", H5T_NATIVE_INT, attr_space,
                                  H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr_id, H5T_NATIVE_INT, &step);
        H5Aclose(attr_id);
        H5Sclose(attr_space);


        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        H5Fclose(file_id);

        printf("PETSc Vec u written to %s\n", FILE_NAME);

        free(global_data);
        free(recvcounts);
        free(displs);
    }
}


PetscInt readh5(Vec u, PetscInt nx, PetscInt ny, const char *filename){
    PetscInt N = nx * ny;
    double *data = (double *)malloc(N * sizeof(double));

	hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

    hid_t dataset_id = H5Dopen(file_id, DATASET_NAME, H5P_DEFAULT);

    H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    
    // Read time step attribute
    // Open the attribute from the dataset, not from the file
    hid_t attr_id = H5Aopen(dataset_id, "step", H5P_DEFAULT);
    int step = 1;
    H5Aread(attr_id, H5T_NATIVE_INT, &step);
    H5Aclose(attr_id);

    H5Dclose(dataset_id);
    H5Fclose(file_id);

    PetscScalar *u_array;
    PetscInt Istart, Iend;
    VecGetOwnershipRange(u, &Istart, &Iend);
    VecGetArray(u, &u_array);

    for (PetscInt Ii = Istart; Ii < Iend; ++Ii) {
        u_array[Ii - Istart] = data[Ii];
    }

    VecRestoreArray(u, &u_array);
    free(data);
    
    return step;
}



int main(int argc, char **args) {
    PetscInitialize(&argc, &args, NULL, NULL);

    PetscInt nx = 10, ny = 10;
    PetscReal t0 = 0.0, T = 1.0, dt = 0.01;
	PetscInt output_interval = 10, restart_interval = 50;

    PetscReal D = 1.0;  // diffusion coefficient
    PetscReal dx, dy;
    PetscInt restart_step = -1;
    
    
	PetscBool flg;
	PetscOptionsGetInt(NULL, NULL, "-n", &nx, NULL);
	ny = nx;
	PetscOptionsGetReal(NULL, NULL, "-dt", &dt, NULL);
	PetscOptionsGetReal(NULL, NULL, "-T", &T, NULL);
	PetscOptionsGetReal(NULL, NULL, "-t0", &t0, &flg);
	PetscOptionsGetInt(NULL, NULL, "-output_interval", &output_interval, NULL);
	PetscOptionsGetInt(NULL, NULL, "-restart_interval", &restart_interval, NULL);
	PetscOptionsGetInt(NULL, NULL, "-restart", &restart_step, &flg);

    
	PetscInt steps = (PetscInt)((T - t0) / dt);
	PetscInt step = (PetscInt)(t0 / dt);
	PetscReal t = step * dt;


    MPI_Comm comm = PETSC_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Grid spacing
    dx = 1.0 / (nx + 1);
    dy = 1.0 / (ny + 1);
    PetscInt N = nx * ny;
    
    //if(

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
    


	if (restart_step >= 0) {
		char h5file[256];
		snprintf(h5file, sizeof(h5file), "restart_t%03d.h5", step);
		step = readh5(u, nx, ny, h5file);

		t = step * dt;
		
		steps = (PetscInt)(T / dt);
		
		if (rank == 0)
		    PetscPrintf(comm, "Restarting from HDF5 file: %s, step: %d, time: %.3f\n", FILE_NAME, step, t);
	} else {
		step = (PetscInt)(t0 / dt);
		t = step * dt;


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
    }


    // Solver
    KSP ksp;
    KSPCreate(comm, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetFromOptions(ksp);

    // Time stepping
    for (; step <= steps; ++step) {
        PetscReal t = step * dt;

        // Assemble RHS
        PetscScalar *u_array;
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
        
        if(step % output_interval == 0) {
            char filename[256];
            snprintf(filename, sizeof(filename), "solution_t%03d.vts", step);
            WriteVTK2D(filename, u, nx, ny, PETSC_COMM_WORLD);
            ComputeL2Error(u, nx, ny, t, PETSC_COMM_WORLD);
        }
        
        if(step % restart_interval == 0) {
            writeh5(u, nx, ny, step);
        }
        
    }



    // Clean up
    KSPDestroy(&ksp);
    VecDestroy(&u);
    VecDestroy(&b);
    MatDestroy(&A);
    PetscFinalize();
    return 0;
}

