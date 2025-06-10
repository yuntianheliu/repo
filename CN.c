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

    Vec u, b, s1, s2, temp;
    Mat A, LHS, RHS;
    KSP ksp;

    // Vectors
    VecCreate(PETSC_COMM_WORLD, &u);
    VecSetSizes(u, PETSC_DECIDE, nx);
    VecSetFromOptions(u);
    VecDuplicate(u, &b);
    VecDuplicate(u, &s1);
    VecDuplicate(u, &s2);
    VecDuplicate(u, &temp);

    // Matrix A (Laplacian)
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, nx, nx);
    MatSetFromOptions(A);
    MatSetUp(A);

    PetscInt i, Istart, Iend;
    MatGetOwnershipRange(A, &Istart, &Iend);

    for (i = Istart; i < Iend; ++i) {
        if (i == 0 || i == nx - 1) {
            MatSetValue(A, i, i, 1.0, INSERT_VALUES); // Dirichlet BC
        } else {
            MatSetValue(A, i, i - 1, 1.0 / (dx * dx), INSERT_VALUES);
            MatSetValue(A, i, i, -2.0 / (dx * dx), INSERT_VALUES);
            MatSetValue(A, i, i + 1, 1.0 / (dx * dx), INSERT_VALUES);
        }
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    // LHS = I - 0.5*dt*D*A
    MatDuplicate(A, MAT_COPY_VALUES, &LHS);
    MatScale(LHS, -0.5 * dt * D);
    for (i = Istart; i < Iend; ++i) {
        MatSetValue(LHS, i, i, 1.0, ADD_VALUES);
    }
    MatAssemblyBegin(LHS, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(LHS, MAT_FINAL_ASSEMBLY);

    // RHS = I + 0.5*dt*D*A
    MatDuplicate(A, MAT_COPY_VALUES, &RHS);
    MatScale(RHS, 0.5 * dt * D);
    for (i = Istart; i < Iend; ++i) {
        MatSetValue(RHS, i, i, 1.0, ADD_VALUES);
    }
    MatAssemblyBegin(RHS, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(RHS, MAT_FINAL_ASSEMBLY);

    // Initial condition
    for (i = Istart; i < Iend; ++i) {
        PetscScalar x = i * dx;
        VecSetValue(u, i, sin(PI * x), INSERT_VALUES);
    }
    VecAssemblyBegin(u);
    VecAssemblyEnd(u);

    // KSP Solver
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, LHS, LHS);
    KSPSetFromOptions(ksp);

    // Time stepping loop
    for (int step = 0; step < nt; ++step) {
        PetscScalar t1 = step * dt;
        PetscScalar t2 = (step + 1) * dt;

        // Compute S(t1)
        for (i = Istart; i < Iend; ++i) {
            if (i == 0 || i == nx - 1) {
                VecSetValue(s1, i, 0.0, INSERT_VALUES);
            } else {
                PetscScalar x = i * dx;
                VecSetValue(s1, i, source_term(x, t1), INSERT_VALUES);
            }
        }

        // Compute S(t2)
        for (i = Istart; i < Iend; ++i) {
            if (i == 0 || i == nx - 1) {
                VecSetValue(s2, i, 0.0, INSERT_VALUES);
            } else {
                PetscScalar x = i * dx;
                VecSetValue(s2, i, source_term(x, t2), INSERT_VALUES);
            }
        }

        VecAssemblyBegin(s1); VecAssemblyEnd(s1);
        VecAssemblyBegin(s2); VecAssemblyEnd(s2);

        // b = RHS * u
        MatMult(RHS, u, b);

        // b += 0.5*dt * (s1 + s2)
        VecWAXPY(temp, 1.0, s1, s2);  // temp = s1 + s2
        VecAXPY(b, 0.5 * dt, temp);

        // Solve LHS * u = b
        KSPSolve(ksp, b, u);
    }

    // Error analysis
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
        PetscPrintf(PETSC_COMM_WORLD, "Crank–Nicolson L1 error at t=%.2f: %.6e\n", T, global_err / nx);

    // Clean up
    VecDestroy(&u); VecDestroy(&b); VecDestroy(&s1); VecDestroy(&s2); VecDestroy(&temp);
    MatDestroy(&A); MatDestroy(&LHS); MatDestroy(&RHS);
    KSPDestroy(&ksp);
    PetscFinalize();
    return 0;
}

