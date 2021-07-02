#ifndef IMEX_UTILS_H
#define IMEX_UTILS_H

#include "HYPRE.h"
#include "mfem.hpp"
#include "IRK_utils.hpp"

#include <mpi.h>
#include <map>
#include <vector>
#include <string>
#include <iostream>

using namespace mfem;
using namespace std;


/** Class holding RK Butcher tableau, and associated data required by
    implicit and explicit splitting. */
class IMEXRKData 
{
public:
    // Implicit Runge Kutta type. Enumeration (s, \sigma, p):
    // - s = number of implicit stages
    // - \sigma = number of explicit stages
    // - p = order
    // In this notation, when s = \sigma, we satisfy (2.3)/(2.4) in
    // Ascher et al., and do not need to compute the final explicit
    // stage. This is represented in the stiffly_accurate boolean.
    enum Type { 
        IMEX111 = 111,
        IMEX121 = 121,
        IMEX122 = 122,
        IMEX222 = 222,
        IMEX232 = 232,
        IMEX233 = 233,
        IMEX443 = 443,
    // ARK ESDIRK-ERK schemes: enumeration (s,p), for total number of
    // stages s.
        ARK43 = -43,
        ARK64 = -64
    };

    IMEXRKData() : s(-1) { };
    IMEXRKData(Type ID_) : ID(ID_) { SetData(); };
    ~IMEXRKData() { };
    
    /// Set explicit RK data
    void SetExplicitData(DenseMatrix Ae_, Vector be_, Vector ce_);
    /// Set implicit RK data
    void SetImplicitData(DenseMatrix Ai_, Vector bi_, Vector ci_, bool esdirk_=false);
    void SetID(Type ID_) { ID=ID_; SetData(); };

    bool esdirk;
    bool stiffly_accurate;
    bool use_final_exp_stage;
    int s;

    DenseMatrix Ai;     // Implicit Butcher matrix
    Vector bi;          // Implicit Butcher tableau weights
    DenseMatrix Ae;     // Explicit Butcher matrix
    Vector be;          // Explicit Butcher tableau weights
    Vector c0;          // Butcher tableau nodes (same for implicit and explicit!)

private:    
    Type ID;
    void SetData();
    void InitData();
};



class IMEXEuler : public ODESolver
{
protected:
    Vector k;
    Vector z;
    IRKOperator *imex;    // Spatial discretization. 

public:
    void Init(IRKOperator &_imex);
    void Step(Vector &x, double &t, double &dt) override;
};


class IMEXEuler2 : public ODESolver
{
protected:
    Vector k;
    Vector z;
    IRKOperator *imex;    // Spatial discretization. 

public:
    void Init(IRKOperator &_imex);
    void Step(Vector &x, double &t, double &dt) override;
};



class IMEXRK222 : public ODESolver
{
protected:
    Vector k1;
    Vector k2;
    Vector ke1;
    Vector ke2;
    IRKOperator *imex;    // Spatial discretization. 
    bool indirectExp;
    bool have_stage1;
    double previous_dt;
public:
    IMEXRK222(bool indirectExp_=false);
    void Init(IRKOperator &_imex);
    void Step(Vector &x, double &t, double &dt) override;
};


class IMEXRK232 : public ODESolver
{
protected:
    Vector k1;
    Vector k2;
    Vector ke1;
    Vector ke2;
    Vector ke3;
    IRKOperator *imex;    // Spatial discretization. 
public:
    void Init(IRKOperator &_imex);
    void Step(Vector &x, double &t, double &dt) override;
};


/** Class for two-part additive IMEX RK method, where explicit and implicit
stage vectors are stored. Assume same abscissae, {c} for both schemes. */
class IMEXRK : public ODESolver
{
protected:

    IMEXRKData tableaux;
    std::vector< Vector *> exp_stages;
    std::vector< Vector *> imp_stages;
    IRKOperator *imex;    // Spatial discretization. 

public:
    IMEXRK(IMEXRKData tableaux_) : ODESolver(), tableaux(tableaux_) { };
    IMEXRK(IMEXRKData::Type type_) : ODESolver(), tableaux(type_) { };
    ~IMEXRK();
    void Init(IRKOperator &_imex);
    void Step(Vector &x, double &t, double &dt) override;
};

/** Class for two-part additive IMEX RK method, where stage solutions are
stored. This requires re-evaluating the implicit and explicit components
regularly, but requires roughly half the storage of IMEXRK. Also may be
simpler to implement extrapolation/initial guesses for stages. */
// class IMEXRK_sol : public ODESolver
// {



// }

#if 1

/** Class holding BDF integrator data. 
    - Setting alpha < 0 defines alpha = 2/(q-1), corresponding to classical
    BDF of order q.
*/
class BDFData 
{
public:

    enum Type {
        BDF1 = 01, BDF2 = 02, BDF3 = 03, BDF4 = 04,
        IMEX_BDF1 = 11, IMEX_BDF2 = 12, IMEX_BDF3 = 13,
        IMEX_BDF4 = 14
    };

    BDFData() { };
    BDFData(Type ID_, double alpha_=-1) : ID(ID_), alpha(alpha_) { SetData(); };
    ~BDFData() { };

    int GetID() { return static_cast<int>(ID); };
    void SetID(Type ID_, double alpha_=-1) {
        ID=ID_;
        alpha = alpha_;
        SetData();
    };
    void Print() {
        std::cout << "q     = " << q << "\n";
        std::cout << "alpha = " << alpha << "\n";
        std::cout << "A:\n";
        A.PrintMatlab();
        std::cout << "Be:\n";
        Be.PrintMatlab();
        std::cout << "Bi:\n";
        Bi.Print();
        std::cout << "z:\n";
        z0.Print();
    };

    double alpha;
    int q;              // Number of previous values stored
    bool shifted_nodes; // false = clssical BDF, true = Polynomial BDF w/ shifted nodes
    DenseMatrix A;      // Previous solution coefficients
    Vector Bi;          // Implicit coefficients
    DenseMatrix Be;     // Explicit coefficients
    Vector z0;


private:    
    Type ID;
    void SetData();
    void InitData();
};


class IMEXBDF : public ODESolver
{
private:
    BDFData data;
    bool recompute_exp;
    bool interpolate;
    int initialized;
    double dt_prev;
    std::vector< Vector*> sols;
    std::vector< Vector*> exp_sols;
    IRKOperator *imex;    // Spatial discretization
    IMEXRK *RKsolver;
    std::vector<double> exp_nodes;

    void AlphaStep(Vector &x, double &t, double &dt);
    void ClassicalStep(Vector &x, double &t, double &dt);
    void ClassicalStepNoStore(Vector &x, double &t, double &dt);

    public:

    IMEXBDF(BDFData data_, bool recompute_exp_=false) :
      ODESolver(), data(data_), recompute_exp(recompute_exp_),
      interpolate(false) { };
    IMEXBDF(BDFData::Type scheme, bool recompute_exp_=false) :
      ODESolver(), recompute_exp(recompute_exp_), interpolate(false)
      { data.SetID(scheme); };
    IMEXBDF(BDFData::Type scheme, double alpha) :
      ODESolver(), interpolate(false), recompute_exp(false)
      { data.SetID(scheme, alpha); };
    ~IMEXBDF();

    void Init(IRKOperator &_imex);
    void Step(Vector &x, double &t, double &dt);
    void InterpolateGuess() {interpolate = true; };
};


#endif

#endif







