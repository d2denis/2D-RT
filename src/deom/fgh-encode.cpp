/**
 * writtened by Zhijun Pan
 * 2018.3.26
 * -------------------
 * FGH-DVR for eigenstate
 * of Morse potential
 */
#include "fgh.hpp"

using namespace std;
using namespace arma;
double dotprod(vec a, vec b){
    double result = 0.0;
    vec::iterator j = b.begin();
    for(vec::iterator i=a.begin();i<a.end();++i){
        result += (*i)*(*j);
        ++j;
    }
    return result;
}

double get_pot_1d (double m0,double lamd,double q1,double q2,double de,double beta,double r0,double r) {
    double morse = (1-exp(-beta*(r-r0)));
    double vqr =  r*(q1+q2*r);
    return de*morse*morse+0.5*lamd*vqr*vqr;//direct solve \Omega_H
}
double get_pot_2d (const double gxy,const double  x,const double  y) {
    return gxy*x*y*(x+y);
}
mat get_ham_1d(Json m,double q1,double q2){
      double  m0 = m["m0"].number_value()*am2unit;
      double  ri = m["ri"].number_value()*ar2unit;
      double  rf = m["rf"].number_value()*ar2unit;
      double  de = m["de"].number_value()*cm2unit;
      double  be = m["be"].number_value()/ar2unit;
      double  re = m["re"].number_value()*ar2unit;
//      double  q1 = m["q1"].number_value();
//      double  q2 = m["q2"].number_value();
      int     nr = m["nr"].int_value();
      int     nk = (nr-1)/2;
      double  dr = (rf-ri)/nr;
      double  dk = 2.0*deom_pi/(rf-ri);
      double ttmp= dk*dk/(2.0*m0);
      mat ham    = zeros<mat>(nr,nr);
      double lamd = m["jdru"][0]["lamd"].number_value()*cm2unit;
      for(int i=0;i<nr;++i){
	  for(int j=0;j<nr;++j){
              for(int k=0;k<nk;++k){
		  double tk = 2.0*deom_pi*(k+1)*(i-j)/nr;
		  ham(i,j) += cos(tk)*ttmp*(k+1)*(k+1)*2.0/nr;
	      }
	  }
	  ham(i,i) += get_pot_1d(m0,lamd,q1,q2,de,be,re,ri+i*dr);
//	  ham(i,i) += get_pot_1d(m0,de,be,re,ri+i*dr);
      }
      return ham;
}
double get_energy(Json m,uword ie){
      double  m0 = m["m0"].number_value()*am2unit;
      double  de = m["de"].number_value()*cm2unit;
      double  be = m["be"].number_value()/ar2unit;
      double  w_0= be*sqrt(2*de/m0);
      double  w_x= be*be*0.5/m0;
      double  Ei = w_0*(ie+0.5)-w_x*(ie+0.5)*(ie+0.5);
      return Ei;
}
void mk_S(umat::row_iterator& h_i,umat::row_iterator& h_j, vec::iterator& val, const mat& ham0,const uvec& i_list,const uvec& nr,const uvec& cum,const int loc,const int nmds){
    uword i = i_list(loc);
    uword i_rank = encode(i_list,cum,nmds);
	    (*h_i) = i_rank;
	    (*h_j) = i_rank;
	    (*val) = ham0(i,i);
	    ++h_i;
	    ++h_j;
	    ++val;
    for(uword j=i+1;j<nr(loc);++j){
	  uvec j_list =i_list;
	  j_list(loc) = j;
	  int j_rank = encode(j_list,cum,nmds);
	    *h_i = i_rank;
	    *h_j = j_rank;
	    *val = ham0(i,j);
	    ++h_i;
	    ++h_j;
	    ++val;
	    *h_i = j_rank;
	    *h_j = i_rank;
	    *val = ham0(i,j);
	    ++h_i;
	    ++h_j;
	    ++val;
    }
}
sp_mat dvr_fgh(const int nmds,const int loc,const uvec& cum, const uvec& nr,const fvec& ri,const fvec& rf,const mat& vmn,const mat& ham0){
    uword ndim = cum(nmds-1);
    uword nnz = ndim*nr(loc);
    umat locations(2,nnz);
    umat::row_iterator h_i = locations.begin_row(0);
    umat::row_iterator h_j = locations.begin_row(1);
    vec values(nnz);
    vec::iterator val= values.begin();
	uvec i_list(nmds);
        for(uword i=0; i<nr(loc); i++){
		i_list(loc) = i;
                uword num,n1,n2;
		switch(nmds){
			case 1:mk_S(h_i,h_j,val,ham0,i_list,nr,cum,loc,nmds);
			       break;

			case 2: num = 1-loc;
			       for(uword k=0; k<nr(num);++k){
				   i_list(num)= k;
				   mk_S(h_i,h_j,val,ham0,i_list,nr,cum,loc,nmds);
			       }
			       break;

			case 3: n1 = (loc+1)%3;
			        n2 = (loc+2)%3;
			       for(uword k1=0; k1<nr(n1);++k1){
			           for(uword k2=0; k2<nr(n2);++k2){
				       i_list(n1)= k1;
				       i_list(n2)= k2;
				       mk_S(h_i,h_j,val,ham0,i_list,nr,cum,loc,nmds);
			           }
			       }
		}
	}
    sp_mat hamt(locations,values);
    return hamt;
}
//Truncate according to <E_n> of each mode
/*
void wave_trun(mat& wavetrun,const mat& wavefun,vec& energr, const vec& energy, const uword nmds, const uword ne,const fvec& Eie,const uvec& nr,const fvec& ri,const fvec& rf,const mat& vmn,const field<mat>& hams){
    if(nmds>3){
	    cout << "too many modes\n";
		 exit(EXIT_SUCCESS);
    }
    ivec label = ones<ivec>(ne);
    uvec cum_prod = zeros<uvec>(nmds);
    cum_prod(0) = nr(nmds-1);
    for (uword m=1;m<nmds;m++)
	    cum_prod(m) = cum_prod(m-1)*nr(nmds-m-1);
// batch insertion to accelerate sparce matrix construct
    for (uword loc=0;loc<nmds;loc++){
	    mat ham0 = hams(loc);
	    sp_mat ham = dvr_fgh(nmds,loc,cum_prod,nr,ri,rf,vmn,ham0);
	    for (uword i=0;i<ne;i++){
		if(label(i)){
		    vec Psi_i =  wavefun.col(i);
		    mat wavepsi = ham*Psi_i;
		    double E_i = dotprod(Psi_i,wavepsi);
		    if (E_i > Eie(loc)*1.001)label(i)=0;
		}
	    }
    }
    int wavecol = 0;
    for(uword i=0; i<ne; i++){
	if(label(i)){
	    wavetrun.insert_cols(wavecol,wavefun.col(i));
	    energr.insert_rows(wavecol,energy.row(i));
	    wavecol++;
	}
    }
    wavetrun.save("wavetrun.dat",raw_ascii);
    energr.save("energr.dat",raw_ascii);
}

void mk_T(umat::row_iterator& h_i,umat::row_iterator& h_j, vec::iterator& val,vec& values, ivec& added, const mat& ham0,const uvec& i_list,const uvec& nr,const uvec& cum,const int loc,const int nmds){
    int i = i_list(loc);
    uword i_rank = encode(i_list,cum,nmds);
    if(added(i_rank) > 0 ){
	    values(added(i_rank)-1) += ham0(i,i);
    }else{
	    (*h_i) = i_rank;
	    (*h_j) = i_rank;
	    (*val) = ham0(i,i);
	    ++h_i;
	    ++h_j;
	    ++val;
	    added(i_rank)=val-values.begin();
    }
    for(int j=i+1;j<nr(loc);++j){
	  uvec j_list =i_list;
	  j_list(loc) = j;
	  int j_rank = encode(j_list,cum,nmds);
	    *h_i = i_rank;
	    *h_j = j_rank;
	    *val = ham0(i,j);
	    ++h_i;
	    ++h_j;
	    ++val;
	    *h_i = j_rank;
	    *h_j = i_rank;
	    *val = ham0(i,j);
	    ++h_i;
	    ++h_j;
	    ++val;
    }
}
//Encode O(nr)^4 times, every steps counts.
sp_mat fgh_dvr(const int ndim,const int nmds,const uvec& nr,const fvec& ri,const fvec& rf,const mat& vmn,const field<mat>& hams){
    if(nmds>3){
	    cout << "too many modes\n";
		 exit(EXIT_SUCCESS);
    }
    uword sum_nr = nr(0);
    uvec cum_prod = zeros<uvec>(nmds);
    cum_prod(0) = nr[nmds-1];
    for (int m=1;m<nmds;m++){
	    cum_prod(m) = cum_prod(m-1)*nr(nmds-m-1);
	    sum_nr += nr(m);
    }
    // batch insertion to accelerate sparce matrix construct
    uword nnz = cum_prod(nmds-1)*(sum_nr-nmds+1);
    cout <<"nonzero element = "<<  nnz << endl;
    umat locations(2,nnz);
    umat::row_iterator h_i = locations.begin_row(0);
    umat::row_iterator h_j = locations.begin_row(1);
    vec values(nnz);
    vec::iterator val= values.begin();
    ivec added = regspace<ivec>(-1,-1,-ndim);
    for (int loc=0;loc<nmds;loc++)
    {
	uvec i_list(nmds);
        for(int i=0; i<nr(loc); i++){
		i_list(loc)	= i;
	        mat ham0 =	hams(loc);
                int num,n1,n2;
		switch(nmds){
			case 1:mk_T(h_i,h_j,val,values,added,ham0,i_list,nr,cum_prod,loc,nmds);
			       break;

			case 2: num = 1-loc;
			       for(int k=0; k<nr(num);++k){
				   i_list(num)= k;
				   mk_T(h_i,h_j,val,values,added,ham0,i_list,nr,cum_prod,loc,nmds);
			       }
			       break;

			case 3: n1 = (loc+1)%3;
			        n2 = (loc+2)%3;
			       for(int k1=0; k1<nr(n1);++k1){
			           for(int k2=0; k2<nr(n2);++k2){
				       i_list(n1)= k1;
				       i_list(n2)= k2;
				       mk_T(h_i,h_j,val,values,added,ham0,i_list,nr,cum_prod,loc,nmds);
			           }
			       }
		}
	}
    }
    sp_mat hamt(locations,values);
    return hamt;
}
*/
void wave_energy(mat& wavetrun, vec& energr, const uword nmds, const uword ne,const fvec& Eie,const uvec& nr,const fvec& ri,const fvec& rf,const mat& vmn,const field<mat>& hams){
    if(nmds>3){
	    cout << "too many modes\n";
		 exit(EXIT_SUCCESS);
    }
    ivec label = ones<ivec>(ne);
    uvec cum_prod = zeros<uvec>(nmds);
    cum_prod(0) = nr(nmds-1);
    for (uword m=1;m<nmds;m++)
	    cum_prod(m) = cum_prod(m-1)*nr(nmds-m-1);
    sp_mat hamt(cum_prod(nmds-1),cum_prod(nmds-1));
    field<sp_mat> Hs(nmds);
// batch insertion to accelerate sparce matrix construct 
    for (uword loc=0;loc<nmds;loc++){
	    mat ham0 = hams(loc);
	    Hs(loc) = dvr_fgh(nmds,loc,cum_prod,nr,ri,rf,vmn,ham0);
	    hamt += Hs(loc);
    }
    mat wavefun;
    vec energy;
    eigs_sym(energy,wavefun,hamt,ne,"sm",1e-9);

    for (uword loc=0;loc<nmds;loc++){
	    for (uword i=0;i<ne;i++){
		if(label(i)){
		    vec Psi_i =  wavefun.col(i);
		    mat wavepsi = Hs(loc)*Psi_i;
		    double E_i = dotprod(Psi_i,wavepsi);
		    if (E_i > Eie(loc)*1.001)label(i)=0;
		}
	    }
    }
    uword wavecol = 0;
    for(uword i=0; i<ne; i++){
	if(label(i)){
	    wavetrun.insert_cols(wavecol,wavefun.col(i));
	    energr.insert_rows(wavecol,energy.row(i));
	    wavecol++;
	}
    }
//    wavetrun.save("wavetrun.dat",raw_ascii);
//    energr.save("energr.dat",raw_ascii);
}
mat dipole(const vec& rx,const mat& wave ){
    uword ndim = wave.n_rows;
    uword ne = wave.n_cols;
    mat dip(ne,ne);
    if(rx.n_rows==ndim){
	 for (uword i=0;i<ne;i++){
	    for (uword j=0;j<ne;j++){
		 double sumdim = 0.0;
		 for (uword n=0;n<ndim;n++){
		    sumdim += rx(n)*wave(n,i)*wave(n,j);
		 }
		 dip(i,j) = sumdim;
	    }
        }
    }else{
        cout << rx.n_rows <<'\t' << ndim << endl;
    }
    return dip;
}
void qmod(cube& q1,cube& q2,const uvec& nr,const fvec& ri,const fvec& rf,const mat& wavefun,const uword nmds){
    uword ndim = wavefun.n_rows;
    uword ne = wavefun.n_cols;
    uvec cum_prod = zeros<uvec>(nmds);
    cum_prod(0) = nr(nmds-1);
    for (uword m=1;m<nmds;m++)
	    cum_prod(m) = cum_prod(m-1)*nr(nmds-m-1);
    q1=zeros<cube>(ne,ne,nmds);
    q2=zeros<cube>(ne,ne,nmds);
    for (uword m=0;m<nmds;m++){
	 vec x(ndim);
	 vec x2(ndim);
	 for (uword i=0;i<ndim;i++){
            uvec i_list = decode(i,cum_prod,nmds);
            double    rx = ri(m)+i_list(m)*(rf(m)-ri(m))/nr(m);
	    x(i) = rx;
	    x2(i)= rx*rx;
	 }
	 q1.slice(m) = dipole(x,wavefun);
//	 q2.slice(m) = dipole(x2,wavefun);
	 q2.slice(m) = q1.slice(m)*q1.slice(m);
    }
}
