/**
 * git clone https://github.com/hou-dao/deom.git
 * ---
 * Written by Houdao Zhang 
 * mailto: houdao@connect.ust.hk
 */
#include <cstdio>
#include <cstdlib>
#include "blockdeom.hpp"

static const int ntMax = 100000000;

static inline bool is_converge (const int n, const cx_cube& r1, const cx_cube &r2, const double err) {
    for (int i=0; i<n; ++i) {
        const double tmp = max(vectorise(abs(r1.slice(i)-r2.slice(i))));
        if (tmp > err) {
            printf("iado=%d, Error greater than %16.6e\n",i,tmp);
            return false;
        }
    }
    return true;
}

void blockdeom::equilibrium (cx_cube& ddos, const double dt, const double err, const int nk) {

    int it = 0;
    bool converge = false;
    cx_cube ddosBackup = zeros<cx_cube>(size(ddos));
    FILE *frho = fopen("equilibrium.log","w");
    while (it < ntMax && !converge) {
        double t = it*dt;
        if (it%nk == 0) {
            fprintf (frho, "%16.6e", t/deom_fs2unit);
            for (int i=0; i<ne; ++i) {
                fprintf(frho, "%16.6e", real(ddos(i,i,0)));
            }
            fprintf(frho, "\n");
            // check
            converge = is_converge (nddo,ddos,ddosBackup,err);
            ddosBackup.slices(0,nddo-1) = ddos.slices(0,nddo-1);
        }
        rk4 (ddos,t,dt);
        it += 1;
    } 

    if (it >= ntMax || !converge) {
        printf ("Fail to reach equilibrium at error %16.6e\n", err);
    } else {
        printf ("Equilibrium reached with %d steps!\n", it);
    } 

    fclose(frho);
}
