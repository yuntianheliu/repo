#include <petscksp.h>
#include <math.h>

int main(int argc, char **args) {
  PetscErrorCode ierr;
  Vec u, u_prev, b;
  Mat A;
  KSP ksp;
  PetscInt n = 100, i;
  PetscReal h, dt = 0.001, T = 0.1, t = 0.0, D = 1.0;
  PetscInt steps = (PetscInt)(T / dt);
  PetscMPIInt rank;
    PetscReal error_norm;

  ierr = PetscInitialize(&argc, &args, NULL, NULL); CHKERRQ(ierr);

  h = 1.0 / (n + 1); // Exclude boundaries

  // Create vectors
  ierr = VecCreate(PETSC_COMM_WORLD, &u); CHKERRQ(ierr);
  ierr = VecSetSizes(u, PETSC_DECIDE, n); CHKERRQ(ierr);
  ierr = VecSetFromOptions(u); CHKERRQ(ierr);
  ierr = VecDuplicate(u, &u_prev); CHKERRQ(ierr);
  ierr = VecDuplicate(u, &b); CHKERRQ(ierr);
  
  PetscInt start, end;
  VecGetOwnershipRange(u, &start, &end);

  // Initial condition: u(x) = sin(pi x)
  PetscScalar *u_array;
  ierr = VecGetArray(u, &u_array); CHKERRQ(ierr);
    for (i = start; i < end; ++i) {
      PetscReal x = (i + 1) * h; // (i+1) since domain excludes boundary
      u_array[i - start] = sin(PETSC_PI * x);  // local index
    }
    VecRestoreArray(u, &u_array);
    
    
    
    //exact solution
    Vec exact;
    VecDuplicate(u, &exact);
    PetscInt istart, iend;
  VecGetOwnershipRange(exact, &istart, &iend);
    for (i = istart; i < iend; i++) {
        PetscReal  xval = (i + 1) * h;
        PetscScalar val = PetscSinReal(PETSC_PI * xval) *
                          PetscExpReal(-D * PetscSqr(PETSC_PI) * T);
        VecSetValue(exact, i, val, INSERT_VALUES);
    }
    VecAssemblyBegin(exact); VecAssemblyEnd(exact);

  // Copy u to u_prev
  ierr = VecCopy(u, u_prev); CHKERRQ(ierr);

  // Create matrix A for (I - dt D A)
  ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
  ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n); CHKERRQ(ierr);
  ierr = MatSetFromOptions(A); CHKERRQ(ierr);
  ierr = MatSetUp(A); CHKERRQ(ierr);

  PetscScalar alpha = D * dt / (h * h);
  for (i = 0; i < n; ++i) {
    if (i > 0) {
      ierr = MatSetValue(A, i, i - 1, -alpha, INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = MatSetValue(A, i, i, 1.0 + 2.0 * alpha, INSERT_VALUES); CHKERRQ(ierr);
    if (i < n - 1) {
      ierr = MatSetValue(A, i, i + 1, -alpha, INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  // Set up linear solver
  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

  // Time stepping loop
  for (int step = 0; step < steps; ++step) {
    t += dt;
    ierr = VecCopy(u_prev, b); CHKERRQ(ierr);         // RHS = previous u
    ierr = KSPSolve(ksp, b, u); CHKERRQ(ierr);        // Solve system
    ierr = VecCopy(u, u_prev); CHKERRQ(ierr);         // Update
  }

  // Print final solution
  ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  
    VecAXPY(u, -1.0, exact); // u = u - exact
    VecNorm(u, NORM_2, &error_norm);
    if (rank == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "L2 error = %g\n", error_norm);
    }

  // Clean up
  ierr = VecDestroy(&u); CHKERRQ(ierr);
  ierr = VecDestroy(&u_prev); CHKERRQ(ierr);
  ierr = VecDestroy(&b); CHKERRQ(ierr);
  ierr = MatDestroy(&A); CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);

  ierr = PetscFinalize(); CHKERRQ(ierr);
  return 0;
}

