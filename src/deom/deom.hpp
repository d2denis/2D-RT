/**
 * git clone https://github.com/hou-dao/deom.git
 * ---
 * Written by Houdao Zhang 
 * mailto: houdao@connect.ust.hk
 */
#ifndef DEOM_H_
#define DEOM_H_

#include "armadillo"
#include "trie.hpp"

#include "deomConst.hpp"
#include "deomSyst.hpp"
#include "deomBath.hpp"
#include "deomHidx.hpp"
#include "deomPulse.hpp"

using namespace std;
using namespace arma;

class deom: public syst, public bath, public hidx {

    public:

        cx_cube ddos1;
        cx_cube ddos2;
        cx_cube ddos3;

        deom (const Json& json): syst (json["syst"]), bath (json["bath"]), hidx (json["hidx"]) {
            ddos1.set_size(nsys,nsys,nmax);
            ddos2.set_size(nsys,nsys,nmax);
            ddos3.set_size(nsys,nsys,nmax);
        }

        deom (const syst& s, const bath& b, const hidx& h): syst (s), bath (b), hidx (h) {
            ddos1.set_size(nsys,nsys,nmax);
            ddos2.set_size(nsys,nsys,nmax);
            ddos3.set_size(nsys,nsys,nmax);
        }

        deom (const deom& rhs): syst(rhs.ham1,rhs.qmd1), 
            bath(rhs.temperature, rhs.modLabel, rhs.coef_lft, rhs.coef_rht, rhs.coef_abs, rhs.expn_gam, rhs.delt_res), 
            hidx(rhs.nind, rhs.lmax, rhs.nmax, rhs.lddo, rhs.nddo, rhs.ferr, rhs.keys, rhs.tree, rhs.expn) {
            ddos1.set_size(nsys,nsys,nmax);
            ddos2.set_size(nsys,nsys,nmax);
            ddos3.set_size(nsys,nsys,nmax);
        }

        ~deom () {}

        void oprAct (cx_cube& d_ddos, const mat& sdip, const cube& pdip, const vec& bdip, const cx_cube& ddos, const char lrc='l');

        void iniHei (cx_cube& d_ddos, const mat& sdip, const cube& pdip, const vec& bdip);

        void remSch (cx_cube& d_ddos, const cx_cube& ddos, const double t);

        void remHei (cx_cube& d_ddos, const cx_cube& ddos, const double t);

        void rem (cx_cube& d_ddos, const cx_cube& ddos, const double t, const ivec& projection);

        void rem (cx_cube& d_ddos, const cx_cube& ddos, const double t, const mat& sdip, const cube& pdip, const vec& bdip, const pulse& p);

        void rem (cx_cube& d_ddos, const cx_cube& ddos, const double t, const mat& sdip, const cube& pdip, const vec& bdip, const pulse& p, const ivec& projection);

        void rem (cx_cube& d_ddos, const cx_cube& ddos, const double t, const char sch_hei = 's') {
            if (sch_hei == 's') {
                remSch (d_ddos, ddos, t);
            } else if (sch_hei == 'h') {
                remHei (d_ddos, ddos, t);
            } else {
                printf("sch_hei is invalid!\n");
            }
        }

        void propagation (cx_cube& ddos, const double dt=0.005, const int nt=1000, const int nk=10);

        void equilibrium (cx_cube& ddos, const double dt=0.005, const double err=2.e-8, const int nk=10);

        inline bool is_valid (const cx_mat& ddo) const {
            return any(abs(vectorise(ddo))>ferr);
        }

        void filter (cx_cube& ddos) {
            int n = 1;
            int l = 0;
            for (int iddo=1; iddo<nddo; ++iddo) {
                TrieNode* p = tree.find(keys(iddo).key);
                if (is_valid(ddos.slice(iddo))) {
                    if (n != iddo) {
                        p->rank = n;
                        keys(n) = keys(iddo);
                        ddos.slice(n) = ddos.slice(iddo);
                    }
                    l = l>(p->tier)?l:(p->tier);
                    ++n;
                } else {
                    p->rank = -9527;
                }
            }
            lddo = l;
            nddo = n;
        }

        template<typename... Tc>
            void rk4 (cx_cube& ddos, const double t, const double dt, const Tc&... args) {

                const double dt2 = dt*0.5;
                const double dt6 = dt/6.0;

                // K1
                const int nddo0 = nddo;
                rem (ddos1,ddos,t,args...);
                ddos3.slices(0,nddo0-1) = ddos.slices(0,nddo0-1)+ddos1.slices(0,nddo0-1)*dt2;
                if (nddo > nddo0) {
                    ddos3.slices(nddo0,nddo-1) = ddos1.slices(nddo0,nddo-1)*dt2;
                }
                // K2
                const int nddo1 = nddo;
                rem (ddos2,ddos3,t+0.5*dt,args...);
                ddos1.slices(0,nddo1-1) += ddos2.slices(0,nddo1-1)*2.0;
                if (nddo > nddo1) {
                    ddos1.slices(nddo1,nddo-1) = ddos2.slices(nddo1,nddo-1)*2.0;
                }
                ddos3.slices(0,nddo0-1) = ddos.slices(0,nddo0-1)+ddos2.slices(0,nddo0-1)*dt2;
                if (nddo > nddo0) {
                    ddos3.slices(nddo0,nddo-1) = ddos2.slices(nddo0,nddo-1)*dt2;
                }
                // K3
                const int nddo2 = nddo;
                rem (ddos2,ddos3,t+0.5*dt,args...);
                ddos1.slices(0,nddo2-1) += ddos2.slices(0,nddo2-1)*2.0;
                if (nddo > nddo2) {
                    ddos1.slices(nddo2,nddo-1) = ddos2.slices(nddo2,nddo-1)*2.0;
                }
                ddos3.slices(0,nddo0-1) = ddos.slices(0,nddo0-1)+ddos2.slices(0,nddo0-1)*dt;
                if (nddo > nddo0) {
                    ddos3.slices(nddo0,nddo-1) = ddos2.slices(nddo0,nddo-1)*dt;
                }
                // K4
                const int nddo3 = nddo;
                rem (ddos2,ddos3,t+dt,args...);
                ddos1.slices(0,nddo3-1) += ddos2.slices(0,nddo3-1);
                if (nddo > nddo3) {
                    ddos1.slices(nddo3,nddo-1) = ddos2.slices(nddo3,nddo-1);
                }
                ddos.slices(0,nddo0-1) += ddos1.slices(0,nddo0-1)*dt6;
                if (nddo > nddo0) {
                    ddos.slices(nddo0,nddo-1) = ddos1.slices(nddo0,nddo-1)*dt6;
                }
                filter (ddos);
            }

        cx_double Trace (const mat& sdip, const cx_cube& ddos) const {
            return trace(sdip*ddos.slice(0));
        }

        cx_double Trace (const mat& sdip, const cube& pdip, const vec& bdip, const cx_cube& ddos) const {
            cx_double result = trace(sdip*ddos.slice(0));
            for (int mp=0; mp<nind; ++mp) {
                const int m = modLabel(mp);
                ivec key0 = zeros<ivec>(mp+1);
                key0(mp) = 1;
                const cx_double sn = bdip(mp)*sqrt(coef_abs(mp));
                TrieNode* p = tree.find(key0); 
                if (p && p->rank>=0) {
                    int loc = p->rank;
                    result += sn*trace(pdip.slice(m)*ddos.slice(loc));
                }
            }       
            return result;
        }
};


void resp1st (const double w_max, const int nt, const double dt,
              const double staticErr, const int nk,
              const mat& sdip, const cube& pdip, const vec& bdip,
              const char& sch_hei, const syst& s, const bath& b, const hidx& h);

void resp2nd (const double w1_max, const double w2_max, const int nt1, const int nt2, const double dt,
              const double staticErr, const int nk,
              const mat& sdip, const cube& pdip, const vec& bdip,
              const char& sch_hei, const syst& s, const bath& b, const hidx& h);

void resp3rd (const double w1_max, const double t2_max, const double w3_max, const int nt1, const int nt2, const int nt3, const double dt,
              const double staticErr, const int nk,
              const mat& sdip, const cube& pdip, const vec& bdip,
              const char& sch_hei, const syst& s, const bath& b, const hidx& h);
              
#endif
