#ifndef IRK_ILU_HPP
#define IRK_ILU_HPP

#include "mfem.hpp"
#include "IRK.hpp"

namespace mfem
{

struct ILU_IRK : ODESolver
{
   enum class Type
   {
      COUPLED,
      UNCOUPLED,
      UNCOUPLED_SHIFTED
   };

   RKData rk_data;
   HypreParMatrix &L;
   std::unique_ptr<HypreParMatrix> J;
   std::unique_ptr<HypreParMatrix> J_prec;
   BlockILU prec;
   GMRESSolver gmres;
   mutable Vector w, b, ws, bs, Lu;

   ILU_IRK(RKData::Type rk_type, HypreParMatrix &M, HypreParMatrix &L_,
           int block_size, double dt, Type prec_type);

   virtual void Step(Vector &u, double &t, double &dt) override;
};

}

#endif
