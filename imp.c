#include <petscksp.h>
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

    PetscInt nx = 100;
    PetscScalar L = 1.0, D = 1.0, dx = L / (nx - 1);
    PetscScalar dt = 0.01, T = 0.1;
    PetscInt nt = (PetscInt)(T / dt);

    Vec u, b;
    Mat A;
    KSP ksp;

    // Create solution and RHS vectors
    VecCreate(PETSC_COMM_WORLD, &u);
    VecSetSizes(u, PETSC_DECIDE, nx);
    VecSetFromOptions(u);
    VecDuplicate(u, &b);

    // Create matrix A for Laplacian
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, nx, nx);
    MatSetFromOptions(A);
    MatSetUp(A);

    PetscInt i, Istart, Iend;
    MatGetOwnershipRange(A, &Istart, &Iend);

    for (i = Istart; i < Iend; ++i) {
        if (i == 0 || i == nx - 1) {
            // Dirichlet BCs
            MatSetValue(A, i, i, 1.0, INSERT_VALUES);
        } else {
            MatSetValue(A, i, i - 1, 1.0 / (dx * dx), INSERT_VALUES);
            MatSetValue(A, i, i, -2.0 / (dx * dx), INSERT_VALUES);
            MatSetValue(A, i, i + 1, 1.0 / (dx * dx), INSERT_VALUES);
        }
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    // Shift matrix for backward Euler: M = I - dt * D * A
    Mat M;
    MatDuplicate(A, MAT_COPY_VALUES, &M);
    MatScale(M, -dt * D);
    for (i = Istart; i < Iend; ++i) {
        MatSetValue(M, i, i, 1.0, ADD_VALUES); // Add identity: I - dt*D*A
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    // Initial condition
    for (i = Istart; i < Iend; ++i) {
        PetscScalar x = i * dx;
        PetscScalar val = sin(PI * x);
        VecSetValue(u, i, val, INSERT_VALUES);
    }
    VecAssemblyBegin(u);
    VecAssemblyEnd(u);

    // Create KSP solver
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, M, M);
    KSPSetFromOptions(ksp);

    // Time stepping
    for (int step = 0; step < nt; ++step) {
        PetscScalar t_next = (step + 1) * dt;

        // Build RHS b = u^n + dt * S(t^{n+1})
        VecCopy(u, b);
        for (i = Istart; i < Iend; ++i) {
            if (i == 0 || i == nx - 1) continue;
            PetscScalar x = i * dx;
            PetscScalar s = dt * source_term(x, t_next);
            VecSetValue(b, i, s, ADD_VALUES);
        }
        VecAssemblyBegin(b);
        VecAssemblyEnd(b);

        // Solve M u^{n+1} = b
        KSPSolve(ksp, b, u);
    }

    // Compute L1 error vs exact
    PetscScalar local_err = 0.0;
    const PetscScalar *u_array;
    VecGetArrayRead(u, &u_array);
    for (i = Istart; i < Iend; ++i) {
        PetscScalar x = i * dx;
        local_err += fabs(u_array[i - Istart] - exact_solution(x, T));
    }
    VecRestoreArrayRead(u, &u_array);

    PetscScalar global_err = 0.0;
    MPI_Allreduce(&local_err, &global_err, 1, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);
    if (Istart == 0)
        PetscPrintf(PETSC_COMM_WORLD, "implicit euler L1 error at t=%.2f: %.6e\n", T, global_err / nx);

    // Cleanup
    VecDestroy(&u);
    VecDestroy(&b);
    MatDestroy(&A);
    MatDestroy(&M);
    KSPDestroy(&ksp);
    PetscFinalize();
    return 0;
}

