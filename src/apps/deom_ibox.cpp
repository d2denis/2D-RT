/**
 * writtened by Zhijun Pan
 * 2018.3.26
 * -------------------
 * FGH-DVR for eigenstate 
 * of Morse potential
 */
#include <ctime>
#include <chrono>
#include <iomanip>

#include <iostream>
#include <sstream>
#include <string>
#include "fgh.hpp"
#include "deom.hpp"

using namespace std;
using namespace arma;
using namespace json11;

mat mattprod(cube a, vec b){
    mat result = zeros<mat>(size(a.slice(0)));
    uword nmds = a.n_slices;
    for(uword i=0;i<nmds;++i){
        result += a.slice(i)*b(i);
    }
    return result;
}
cube cubeprod(cube a, vec b){
    cube result = zeros<cube>(size(a));
    uword nmds = a.n_slices; 
    for(uword i=0;i<nmds;++i){
        result.slice(i) = a.slice(i)*b(i);
    }
    return result;
}
int main() {

    ifstream jsonFile("morse.json");
    stringstream strStream;
    strStream << jsonFile.rdbuf();
    string jsonStr = strStream.str();
    string err;

    const Json json = Json::parse(jsonStr,err);
    if (!err.empty()) {
        printf ("Error in parsing input file: %s\n", err.c_str());
        return 0;
    }

    const uword    ne = json["ne"].int_value();
    const uword    ie = json["ie"].int_value();
    const Json  mds = json["modes"];
    uword  nmds = 0;
    uword  ndim = 1;
    uvec nr (mmax);
    fvec ri (mmax);
    fvec rf (mmax);
    fvec Eie(mmax); 
    vec q1(mmax);
    vec q2(mmax);
    field<mat> hams(mmax);   // field for different nr in modes
    for (uword nmode=0;!mds[nmode].is_null();nmode++){
      nr[nmode] = mds[nmode]["nr"].int_value();
      ri[nmode] = mds[nmode]["ri"].number_value()*ar2unit;
      rf[nmode] = mds[nmode]["rf"].number_value()*ar2unit;
      q1[nmode] = mds[nmode]["q1"].number_value();
      q2[nmode] = mds[nmode]["q2"].number_value();
      ndim *= nr[nmode];
      hams(nmode) = get_ham_1d(mds[nmode],q1[nmode],q2[nmode]); //immutable Json, ar2unit don't affect ri,rf
      Eie(nmode)  = get_energy(mds[nmode],ie);
      nmds++;
    }
    cout << "there are " << nmds << " modes" << '\n';
    mat vmn = zeros<mat>(nmds,nmds);// the couple coefficient, not consider now
    printime("Begin calculation wavefunction and energy:");
    mat wavefun;
    vec energy ;
    wave_energy( wavefun, energy, nmds, ne,Eie,nr,ri, rf,vmn,hams);
    printime("End loading wavefunction and energy:");

    vec d1(nmds);
    vec d2(nmds);
    vec lamd(nmds);
    vec gamd(nmds);
    for (uword nmode=0;nmode<nmds;nmode++){
      d1[nmode] = mds[nmode]["d1"].number_value();
      d2[nmode] = mds[nmode]["d2"].number_value();
//      q1[nmode] = mds[nmode]["q1"].number_value();
//      q2[nmode] = mds[nmode]["q2"].number_value();
      lamd[nmode] = mds[nmode]["jdru"][0]["lamd"].number_value()*cm2unit;
      gamd[nmode] = mds[nmode]["jdru"][0]["gamd"].number_value()*cm2unit;
    }
    ifstream jsonFile2("setting.json");
    stringstream strStream2;
    strStream2 << jsonFile2.rdbuf();
    string jsonStr2 = strStream2.str();

    const Json sett = Json::parse(jsonStr2,err);
    if (!err.empty()) {
        printf ("Error in parsing input file: %s\n", err.c_str());
        return 0;
    }
//    const Json  mysyst = deft["syst"];
    const Json  mybath = sett["bath"];
    const Json  myhidx = sett["hidx"];
    int npsd = mybath["npsd"].int_value();
    int pade = mybath["pade"].int_value();
    double temp = mybath["temp"].number_value()*kt2unit;
    int lmax = myhidx["lmax"].int_value();
    int nmax = myhidx["nmax"].int_value();
    double ferr = myhidx["ferr"].number_value();
    cx_mat hm = deom_c1*diagmat(energy);
    cube dip,pol;
    qmod(dip,pol,nr,ri,rf,wavefun,nmds);
//    vec fw_im=fit1d(hm,dip,pol,d1,d2,q1,q2,lamd,gamd);
//    fw_im.save("fit1d.dat",raw_ascii);
    cx_cube qm =deom_c1*(cubeprod(dip,q1)+cubeprod(pol,q2));
    syst s(hm,qm);
    hm.save("inp_hams.mat",arma_ascii);
    qm.save("inp_qmds.mat",arma_ascii);
    bath b(npsd,pade,temp,lamd,gamd);
    b.delt_res.print("delta_residual:");
    cx_vec index = b.expn_gam;
    hidx h(lmax,nmax,ferr,index);
	    
    const double w1_max = sett["spec"]["w1max"].number_value()*cm2unit;              
    const int    nt1 = 400;//sett["spec"]["nt1"].int_value();
    const double dt = sett["spec"]["dt"].number_value()*fs2unit;
    const double staticErr = sett["spec"]["staticErr"].number_value();
    const int    nk = 32;//sett["spec"]["nk"].int_value();
    const string sch_hei = "s";//sett["spec"]["sch_hei"].string_value();

    mat  sdip = mattprod(dip,d1)+mattprod(pol,d2);;
    sdip.print("sdip:");
    hm.print("Hamitonian:");
    qm.print("Q mode:");
    cube pdip = zeros<cube>(size(qm));
    vec  bdip = zeros<vec>(size(index));
    resp1st (w1_max, nt1, dt, staticErr, nk, sdip, pdip, bdip, sch_hei[0], s, b, h);
    return 0;
}
