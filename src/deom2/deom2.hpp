/**
 * git clone https://github.com/hou-dao/deom.git
 * ---
 * Written by Houdao Zhang 
 * mailto: houdao@connect.ust.hk
 */
#ifndef DEOM2_H_
#define DEOM2_H_

#include "armadillo"
#include "trie.hpp"

#include "deomConst.hpp"
#include "deomSyst2.hpp"
#include "deomBath2.hpp"
#include "deomHidx.hpp"

using namespace std;
using namespace arma;

class deom2: public syst, public bath, public hidx {

    public:

        cx_cube ddos1;
        cx_cube ddos2;
        cx_cube ddos3;

        deom2 (const Json& json): syst (json["syst"]), bath (json["bath"]), hidx (json["hidx"]) {
            ddos1.set_size(nsys,nsys,nmax);
            ddos2.set_size(nsys,nsys,nmax);
            ddos3.set_size(nsys,nsys,nmax);
        }

        deom2 (const syst& s, const bath& b, const hidx& h): syst (s), bath (b), hidx (h) {
            ddos1.set_size(nsys,nsys,nmax);
            ddos2.set_size(nsys,nsys,nmax);
            ddos3.set_size(nsys,nsys,nmax);
        }

        deom2 (const deom2& rhs): syst(rhs.ham1,rhs.qmd1,rhs.qmd2), 
            bath(rhs.temperature,rhs.modLabel,rhs.coef_lft,rhs.coef_rht,rhs.coef_abs,rhs.expn_gam,rhs.delt_res,rhs.alpha1,rhs.alpha2), 
            hidx(rhs.nind, rhs.lmax, rhs.nmax, rhs.lddo, rhs.nddo, rhs.ferr, rhs.keys, rhs.tree, rhs.expn) {
            ddos1.set_size(nsys,nsys,nmax);
            ddos2.set_size(nsys,nsys,nmax);
            ddos3.set_size(nsys,nsys,nmax);
        }

        ~deom2 () {}

        void rem (cx_cube& d_ddos, const cx_cube& ddos, const double t);

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
};

#endif
