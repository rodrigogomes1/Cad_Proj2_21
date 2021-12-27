/**
 * Simulation over time of the N-Body problem
 * We don't pretend to follow Physics laws,
 * just some computation workload to play with.
 *
 * Based on the Basic N-Body solver from:
 * An introduction to parallel programming, 2nd Edition
 * Peter Pacheco, Matthew Malensek
 *
 * Rodrigo Gomes FCT/UNL 2021
 * Ruben Vaz
 * CAD - 2021/2022
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

//#define N_BODIES    2048
#define N_BODIES 4		// good for development and testing
#define Gconst  0.1

double mas[N_BODIES];
double posx[N_BODIES];  /* just a 2D simulation */
double posy[N_BODIES];
double velx[N_BODIES];
double vely[N_BODIES];
double forcex[N_BODIES];
double forcey[N_BODIES];

double localVelX[1];
double localVelY[1];

void initParticles() {
    for (int p=0; p<N_BODIES; p++) {
        mas[p]=1;                 /* 1 mass unit */
        posx[p]=drand48()*100;    /* 100x100 space */
        posy[p]=drand48()*100;
        velx[p]=0;
        vely[p]=0;
    }
}

void printParticles(FILE *f) {
    // print particles to text file f
    fprintf(f,    "#  pos     vel\n");
    for (int p=0; p<N_BODIES; p++) {
        fprintf(f,"%d (%g,%g) (%g,%g)\n",
               p, posx[p], posy[p], velx[p], vely[p]);
    }
}


void computeForces(int q) {
    // based on the basic solver from book (not the reduced solver)

    for (int k=0; k<N_BODIES; k++) {
        if (k == q) continue; // ignore itself
        double xdiff=posx[q] - posx[k];
        double ydiff=posy[q] - posy[k];
        double dist=sqrt(xdiff*xdiff+ydiff*ydiff);
        double distCub=dist*dist*dist;
        forcex[q] -= Gconst*mas[q]*mas[k]/distCub * xdiff;
        forcey[q] -= Gconst*mas[q]*mas[k]/distCub * ydiff;
    }

}

void moveParticle(int q, double deltat) {
    posx[q] += deltat*localVelX[q];
    posy[q] += deltat*localVelY[q];
    localVelX[q] += deltat/mas[q] * forcex[q];
    localVelY[q] += deltat/mas[q] * forcey[q];
}

void simulateStep(double deltat,int loc_pos, int loc_n) {
    memset(forcex, 0, sizeof forcex);
    memset(forcey, 0, sizeof forcey);

    for ( int q=loc_pos; q<loc_pos+loc_n; q++)
        computeForces(q);
    for (int q=loc_pos; q<loc_pos+loc_n; q++)
        moveParticle(q, deltat);
}


int main(int argc, char *argv[]) {
    int nSteps = 100;  // default (you can give this at the command line)
    double time = 100;

    if (argc==2) {
        nSteps = atoi(argv[1]);		// number of steps
    }
    double deltat = time/nSteps;

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD , &size);
    MPI_Comm_rank(MPI_COMM_WORLD , &rank);
    printf("Mpi started from %d of %d\n", rank, size);



    int loc_n= N_BODIES/size;
    int loc_pos=loc_n*rank;

    clock_t t;

    if(rank==0){
        printf("Started %d steps!\n", nSteps);
        printf("Passou \n");
        initParticles();
        t= clock();
    }


    MPI_Bcast(mas,N_BODIES,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(posx,N_BODIES,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(posy,N_BODIES,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Scatter(velx,loc_n,MPI_DOUBLE,localVelX, loc_n,MPI_DOUBLE,0, MPI_COMM_WORLD);
    MPI_Scatter(vely,loc_n,MPI_DOUBLE,localVelY, loc_n,MPI_DOUBLE,0, MPI_COMM_WORLD);


    for (int s=0; s< nSteps; s++){
        simulateStep(deltat,loc_pos, loc_n);
        MPI_Allgather(MPI_IN_PLACE,loc_n,MPI_DOUBLE, posx,loc_n,MPI_DOUBLE,MPI_COMM_WORLD);
        MPI_Allgather(MPI_IN_PLACE,loc_n,MPI_DOUBLE, posy,loc_n,MPI_DOUBLE,MPI_COMM_WORLD);
    }
    for(int i=0; i< N_BODIES; i++){
        printf("localVelX %f in %d with index  %d\n", localVelX[i],rank,i);
    }


    MPI_Gather(localVelX,loc_n,MPI_DOUBLE,velx,loc_n,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Gather(localVelY,loc_n,MPI_DOUBLE,vely,loc_n,MPI_DOUBLE,0,MPI_COMM_WORLD);

    for(int i=0; i< N_BODIES; i++){
        //printf("VelX %f in %d with index  %d\n", velx[i],rank,i);
    }


    if(rank==0){
        t = clock()-t;

        printParticles(stdout);		// check if this solution is correct
        printf("time: %f s\n", t/(double)CLOCKS_PER_SEC);
    }



    MPI_Finalize();

    return 0;
}
