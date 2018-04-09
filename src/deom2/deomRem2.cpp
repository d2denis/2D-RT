/**
 * git clone https://github.com/hou-dao/deom.git
 * ---
 * Written by Houdao Zhang 
 * mailto: houdao@connect.ust.hk
 */
#include <cmath>
#include "deom2.hpp"

void deom2::rem (cx_cube& dtotal, const cx_cube& total, const double t) {

    const int nsav = nddo;
    dtotal.slices(0,nddo-1).zeros();

    cx_cube qddo1(size(qmd1)); 
    cx_cube ddoq1(size(qmd1)); 
    cx_cube qddo2(size(qmd2)); 
    cx_cube ddoq2(size(qmd2)); 

    for (int iado=0; iado<nsav; ++iado) {
        const cx_mat& ado = total.slice(iado);
        if (iado==0 || is_valid (ado)) {
            const hnod& nod = keys(iado);
            ivec key0(nod.key);
            int tier = tree.find(key0)->tier;

            dtotal.slice(iado) += -deom_ci*(ham1*ado-ado*ham1)-nod.gams*ado;
            for (int m=0; m<nmod; ++m) {
                qddo1.slice(m) = qmd1.slice(m)*ado;
                ddoq1.slice(m) = ado*qmd1.slice(m);
                qddo2.slice(m) = qmd2.slice(m)*ado;
                ddoq2.slice(m) = ado*qmd2.slice(m);
                if (abs(delt_res(m)) > 1.e-15) {
                    dtotal.slice(iado) -= delt_res(m)*(
                        qmd1.slice(m)*(qddo1.slice(m)-ddoq1.slice(m))
                       -(qddo1.slice(m)-ddoq1.slice(m))*qmd1.slice(m));
                }
            }

            for (int mp=0; mp<nind; ++mp) { // contribute to same tier
                ivec key1 = gen_key(key0, mp, 1);
                const int m = modLabel(mp);
                const int k = key1(mp)-1;
                for (int nq=0; nq<nind; ++nq) {
                    ivec key2 = gen_key(key1, nq,-1);
                    if (!key2.is_empty()) {
                        // const int n = modLabel(nq);
                        const int l = (key2.n_rows<(unsigned int)(nq+1))?1:(key2(nq)+1);
                        const cx_double sn = -deom_ci*2.0*alpha2(m)*sqrt((k+1)/coef_abs(mp)*l*coef_abs(nq));
                        const cx_double cl = sn*coef_lft(mp);
                        const cx_double cr = sn*coef_rht(mp);
                        if (!tree.try_insert(key2,nddo)) {
                            int loc = tree.find(key2)->rank;
                            dtotal.slice(loc) += cl*qddo2.slice(m)-cr*ddoq2.slice(m);
                        } else {
                            keys(nddo) = hnod(nod.gams+expn(mp)-expn(nq),key2);
                            dtotal.slice(nddo) = cl*qddo2.slice(m)-cr*ddoq2.slice(m);
                            nddo += 1;
                        }
                    }
                }
            }

            if (tier < lmax-1) { // contribute to tier+2
                for (int mp=0; mp<nind; ++mp) {
                    ivec key1 = gen_key(key0, mp, 1);
                    const int m = modLabel(mp);
                    const int k = key1(mp)-1;
                    for (int nq=0; nq<nind; ++nq) {
                        ivec key2 = gen_key(key1, nq, 1);
                        // const int n = modLabel(nq);
                        const int l = (mp==nq)?(key2(nq)-2):(key2(nq)-1);
                        const cx_double sn = -deom_ci*alpha2(m)*sqrt((k+1)/coef_abs(mp)*(l+1)/coef_abs(nq));
                        const cx_double cl = sn*coef_lft(mp)*coef_lft(nq);
                        const cx_double cr = sn*coef_rht(mp)*coef_rht(nq);
                        if (!tree.try_insert(key2,nddo)) {
                            int loc = tree.find(key2)->rank;
                            dtotal.slice(loc) += cl*qddo2.slice(m)-cr*ddoq2.slice(m);
                        } else {
                            keys(nddo) = hnod(nod.gams+expn(mp)+expn(nq),key2);
                            dtotal.slice(nddo) = cl*qddo2.slice(m)-cr*ddoq2.slice(m);
                            nddo += 1;
                        }
                    }
                }
            }

            for (int mp=0; mp<nind; ++mp) { // contribute to tier-2
                ivec key1 = gen_key(key0, mp,-1);
                if (!key1.is_empty()) {
                    const int m = modLabel(mp);
                    const int k = (key1.n_rows<(unsigned int)(mp+1))?1:(key1(mp)+1);
                    for (int nq=0; nq<nind; ++nq) {
                        ivec key2 = gen_key(key1, nq,-1);
                        if (!key2.empty()) {
                            // const int n = modLabel(nq);
                            const int l = (key2.n_rows<(unsigned int)(nq+1))?1:(key2(nq)+1);
                            const cx_double sn = -deom_ci*alpha2(m)*sqrt(k*coef_abs(mp)*l*coef_abs(nq));
                            if (!tree.try_insert(key2,nddo)) {
                                int loc = tree.find(key2)->rank;
                                dtotal.slice(loc) += sn*(qddo2.slice(m)-ddoq2.slice(m));
                            } else {
                                keys(nddo) = hnod(nod.gams-expn(mp)-expn(nq),key2);
                                dtotal.slice(nddo) = sn*(qddo2.slice(m)-ddoq2.slice(m));
                                nddo += 1;
                            }
                        }
                    }
                }
            }

            if (tier < lmax) {
                for (int mp=0; mp<nind; ++mp) {
                    ivec key1 = gen_key(key0, mp, 1);
                    const int m = modLabel(mp);
                    const int n = key1(mp)-1;
                    const cx_double sn = -deom_ci*alpha1(m)*sqrt((n+1)/coef_abs(mp));
                    const cx_double cl = sn*coef_lft(mp);
                    const cx_double cr = sn*coef_rht(mp);
                    if (!tree.try_insert(key1,nddo)) {
                        int loc = tree.find(key1)->rank;
                        dtotal.slice(loc) += cl*qddo1.slice(m)-cr*ddoq1.slice(m);
                    } else {
                        keys(nddo) = hnod(nod.gams+expn(mp),key1);
                        dtotal.slice(nddo) = cl*qddo1.slice(m)-cr*ddoq1.slice(m);
                        nddo += 1;
                    }
                }
            }

            for (int mp=0; mp<nind; ++mp) {
                ivec key1 = gen_key(key0, mp, -1);
                if (!key1.is_empty()) {
                    const int m = modLabel(mp);
                    const int n = (key1.n_rows<(unsigned int)(mp+1))?1:(key1(mp)+1);
                    const cx_double sn = -deom_ci*alpha1(m)*sqrt(n*coef_abs(mp));
                    if (!tree.try_insert(key1,nddo)) {
                        int loc = tree.find(key1)->rank;
                        dtotal.slice(loc) += sn*(qddo1.slice(m)-ddoq1.slice(m)); 
                    } else {
                        keys(nddo) = hnod(nod.gams-expn(mp),key1);
                        dtotal.slice(nddo) = sn*(qddo1.slice(m)-ddoq1.slice(m)); 
                        nddo += 1;
                    }
                }
            }
        }
    }
}
