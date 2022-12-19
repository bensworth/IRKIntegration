#include "IRK_utils.hpp"
#include <iostream>
#include <fstream>
#include <map>
#include <iomanip> 
#include <cmath> 


// -------------------------------------------------------------------- //
// -------------------------------------------------------------------- //

/// Kronecker transform between two block vectors: y <- (A \otimes I)*x
void KronTransform(const DenseMatrix &A, const BlockVector &x, BlockVector &y)
{
    for (int block = 0; block < A.Height(); block++) {
        int j = 0;
        if (fabs(A(block,j)) > 0.) {
            y.GetBlock(block).Set(A(block,j), x.GetBlock(j));
        }
        else {
            y.GetBlock(block) = 0.;
        }
        for (int j = 1; j < A.Width(); j++) {
            if (fabs(A(block,j)) > 0.) y.GetBlock(block).Add(A(block,j), x.GetBlock(j));
        }
    }
};

/// Kronecker transform between two block vectors using A transpose: y <- (A^T \otimes I)*x
void KronTransformTranspose(const DenseMatrix &A, const BlockVector &x, BlockVector &y)
{
    for (int block = 0; block < A.Width(); block++) {
        int i = 0;
        if (fabs(A(i,block)) > 0.) {
            y.GetBlock(block).Set(A(i,block), x.GetBlock(i));
        }
        else {
            y.GetBlock(block) = 0.;
        }
        for (int i = 1; i < A.Height(); i++) {
            if (fabs(A(i,block)) > 0.) y.GetBlock(block).Add(A(i,block), x.GetBlock(i));
        }
    }
};

/// Kronecker transform in place: x <- (A \otimes I)*x
void KronTransform(const DenseMatrix &A, BlockVector &x)
{
    BlockVector y(x);
    for (int block = 0; block < A.Height(); block++) {
        int j = 0;
        if (fabs(A(block,j)) > 0.) {
            x.GetBlock(block).Set(A(block,j), y.GetBlock(j));
        }
        else {
            x.GetBlock(block) = 0.;
        }
        for (int j = 1; j < A.Width(); j++) {
            if (fabs(A(block,j)) > 0.) x.GetBlock(block).Add(A(block,j), y.GetBlock(j));
        }
    }
};

/// Kronecker transform using A transpose in place: x <- (A^T \otimes I)*x
void KronTransformTranspose(const DenseMatrix &A, BlockVector &x)
{
    BlockVector y(x);
    for (int block = 0; block < A.Width(); block++) {
        int i = 0;
        if (fabs(A(i,block)) > 0.) {
            x.GetBlock(block).Set(A(i,block), y.GetBlock(i));
        }
        else {
            x.GetBlock(block) = 0.;
        }
        for (int i = 1; i < A.Height(); i++) {
            if (fabs(A(i,block)) > 0.) x.GetBlock(block).Add(A(i,block), y.GetBlock(i));
        }
    }
};

// -------------------------------------------------------------------- //
// -------------------------------------------------------------------- //

IRKOperator::IRKOperator(MPI_Comm comm, bool linearly_imp, int n,
    double t, Type type, ExplicitGradients ex_gradients) 
    : TimeDependentOperator(n, t, type), m_comm(comm),
    m_gradients(ex_gradients), temp(n), m_linearly_imp(linearly_imp)
{
    if (m_linearly_imp) m_gradients = APPROXIMATE;
};

void IRKOperator::ImplicitPrec(const Vector &x, Vector &y) const 
{
    mfem_error("IRKOperator::ImplicitPrec() is not overridden!");
}

void IRKOperator::ImplicitPrec(int index, const Vector &x, Vector &y) const 
{
    mfem_error("IRKOperator::ImplicitPrec() is not overridden!");
}

void IRKOperator::MassMult(const Vector &x, Vector &y) const
{
    mfem_error("IRKOperator::MassMult() is not overridden!");
}

void IRKOperator::MassInv(const Vector &x, Vector &y) const
{
    mfem_error("IRKOperator::MassInv() is not overridden!");
}

void IRKOperator::ExplicitMult(const Vector &x, Vector &y) const
{
    mfem_error("IRKOperator::ExplicitMult() is not overridden!");
}

void IRKOperator::SetExplicitGradient(const Vector &u, double dt, 
                                 const BlockVector &x, const Vector &c)
{
    MFEM_ASSERT(m_gradients == ExplicitGradients::APPROXIMATE, 
                "IRKOperator::SetExplicitGradient() applies only for \
                ExplicitGradients::APPROXIMATE");
    mfem_error("IRKOperator::SetExplicitGradient() is not overridden \
                for IRK methods!");
}

void IRKOperator::SetExplicitGradient(double r, 
                                 const BlockVector &x, const Vector &z)
{
    MFEM_ASSERT(m_gradients == ExplicitGradients::APPROXIMATE, 
                "IRKOperator::SetExplicitGradient() applies only for \
                ExplicitGradients::APPROXIMATE");
    mfem_error("IRKOperator::SetExplicitGradient() is not overridden \
                for PolyIMEX methods!");
}

void IRKOperator::ExplicitGradientMult(const Vector &x, Vector &y) const
{
    MFEM_ASSERT(m_gradients == ExplicitGradients::APPROXIMATE, 
                "IRKOperator::ExplicitGradientMult() applies only for \
                ExplicitGradients::APPROXIMATE");
    mfem_error("IRKOperator::ExplicitGradientMult() is not overridden!");
}

void IRKOperator::SetPreconditioner(int index, double dt, double gamma, int type) 
{
    MFEM_ASSERT(m_gradients == ExplicitGradients::APPROXIMATE, 
                "IRKOperator::SetExplicitGradient() applies only for \
                ExplicitGradients::APPROXIMATE");
    mfem_error("IRKOperator::SetPreconditioner() is not overridden!");
}
    
void IRKOperator::SetExplicitGradients(const Vector &u, double dt, 
                                  const BlockVector &x, const Vector &c)
{
    MFEM_ASSERT(m_gradients == ExplicitGradients::EXACT, 
                "IRKOperator::SetExplicitGradients() applies only for \
                ExplicitGradients::EXACT");
    mfem_error("IRKOperator::SetExplicitGradients() is not overridden \
                for IRK methods!");
}

void IRKOperator::SetExplicitGradients(double r, 
                                  const BlockVector &x, const Vector &z)
{
    MFEM_ASSERT(m_gradients == ExplicitGradients::EXACT, 
                "IRKOperator::SetExplicitGradients() applies only for \
                ExplicitGradients::EXACT");
    mfem_error("IRKOperator::SetExplicitGradients() is not overridden \
                for PolyIMEX methods!");
}

void IRKOperator::ExplicitGradientMult(int index, const Vector &x, Vector &y) const
{
    MFEM_ASSERT(m_gradients == ExplicitGradients::EXACT, 
                "IRKOperator::ExplicitGradientMult() applies only for \
                ExplicitGradients::EXACT");
    mfem_error("IRKOperator::ExplicitGradientMult() is not overridden!");
}

void IRKOperator::SetPreconditioner(int index, double dt, double gamma, Vector weights) 
{
    mfem_error("IRKOperator::SetPreconditioner() is not overridden!");
}

void IRKOperator::AddExplicitGradientsMult(double c, const Vector &weights, 
                                     const Vector &x, Vector &y) const 
{
    MFEM_ASSERT(m_gradients == ExplicitGradients::EXACT, 
                "IRKOperator::AddExplicitGradientsMult() applies only for \
                ExplicitGradients::EXACT");
    
    MFEM_ASSERT(weights.Size() > 0, 
        "IRKOperator::AddExplicitGradientsMult() not defined for empty weights");
        
    for (int i = 0; i < weights.Size(); i++) {
        if (fabs(c*weights(i)) > 0.) {
            ExplicitGradientMult(i, x, temp);
            y.Add(c*weights(i), temp);
        }
    }
}

void IRKOperator::AddExplicitGradientsMult(double c1, const Vector &weights1, 
                                     double c2, const Vector &weights2, 
                                     const Vector &x, 
                                     Vector &y1, Vector &y2) const 
{
    MFEM_ASSERT(m_gradients == ExplicitGradients::EXACT, 
                "IRKOperator::AddExplicitGradientsMult() applies only for \
                ExplicitGradients::EXACT");
    MFEM_ASSERT(weights1.Size() > 0 && weights2.Size() > 0, 
        "IRKOperator::AddExplicitGradientsMult() not defined for empty weights");
    MFEM_ASSERT(weights1.Size() == weights2.Size(), 
        "IRKOperator::AddExplicitGradientsMult() weight vectors need to be of equal length");    
        
    for (int i = 0; i < weights1.Size(); i++) {
        if (fabs(c1*weights1(i)) > 0. || fabs(c2*weights2(i)) > 0.) {
            ExplicitGradientMult(i, x, temp);
            if (fabs(c1*weights1(i)) > 0.) y1.Add(c1*weights1(i), temp);
            if (fabs(c2*weights2(i)) > 0.) y2.Add(c2*weights2(i), temp);
        }
    }
}

// -------------------------------------------------------------------- //
// -------------------------------------------------------------------- //

void IRKStageOper::SetParameters(const Vector *u_, double t_, double dt_)
{ 
    t = t_;
    dt = dt_;
    u = u_;
    getGradientCalls = 0; // Reset counter
};

void IRKStageOper::SetParameters(double t_, double dt_)
{ 
    t = t_;
    dt = dt_;
    u = NULL;
    getGradientCalls = 0; // Reset counter
};

Operator& IRKStageOper::GetGradient(const Vector &w) const
{
    // Update `current_iterate` so that its data points to the current iterate's
    current_iterate.Update(w.GetData(), offsets);
    
    // Increment counter
    getGradientCalls++;
        
    // To stop compiler complaining of no return value    
    return *dummy_gradient;
}

void IRKStageOper::Mult(const Vector &w_vector, Vector &y_vector) const
{
    if (!Butcher.IsIMEX()) {
        MFEM_ASSERT(u, "IRKStageOper::Mult() Requires states to be set, see SetParameters()");
    }

    // Wrap scalar Vectors with BlockVectors
    w_block.Update(w_vector.GetData(), offsets);
    y_block.Update(y_vector.GetData(), offsets);
    
    /* y <- inv(A0)*M*w */
    // MASS MATRIX
    if (IRKOper->isImplicit()) {
        // temp <- M*w
        for (int i = 0; i < Butcher.s; i++) {
            IRKOper->MassMult(w_block.GetBlock(i), temp_block.GetBlock(i));
        }
        KronTransform(Butcher.invA0, temp_block, y_block); // y <- inv(A0)*temp
    // NO MASS MATRIX
    }
    else {
        KronTransform(Butcher.invA0, w_block, y_block); // y <- inv(A0)*w
    }
    
    /* y <- y - N(u + dt*w), where w = (A0 x I)k, for stage vectors k */
    for (int i = 0; i < Butcher.s; i++) { 
        // If u is defined in class, apply operator at u + dt*w (IRK) 
        if (u) {
            add(*u, dt, w_block.GetBlock(i), temp_vector1); // temp1 <- u+dt*w(i)
            IRKOper->SetTime(t + Butcher.c0[i]*dt);
            IRKOper->ImplicitMult(temp_vector1, temp_vector2); // temp2 <- N(temp1, t)
            y_block.GetBlock(i).Add(-1., temp_vector2);
        }
        // If u is not defined in class, apply operator to w (PolyIMEX) 
        else{
            IRKOper->SetTime(t + Butcher.z0[i]*dt);     // TODO : what is correct time??
            IRKOper->ImplicitMult(w_block.GetBlock(i), temp_vector2); // temp2 <- N(temp1, t)
            y_block.GetBlock(i).Add(-1.*dt, temp_vector2);
        }
    }
}

// -------------------------------------------------------------------- //
// -------------------------------------------------------------------- //

QuasiMatrixProduct::QuasiMatrixProduct(DenseMatrix Q) 
    : Array2D<Vector*>(Q.Height(), Q.Height()), height{Q.Height()} 
{
    MFEM_ASSERT(Q.Height() == Q.Width(), "QuasiMatrixProduct:: Matrix must be square");
    
    // Create Vectors of coefficients
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < height; col++) {
            (*this)(row, col) = new Vector(height);
            for (int i = 0; i < height; i++) {
                *((*this)(row, col))[i] = Q(i,row)*Q(i,col); 
            }
        }
    }
}

void QuasiMatrixProduct::Sparsify(int sparsity)
{
    // Sparsify if need be
    switch (sparsity) {
        case 0:
            this->Lump();
            break;
        case 1:
            this->TruncateOffDiags();
            break;
        default:
            break;
    }        
}

void QuasiMatrixProduct::Print() const
{
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < height; col++) {
            mfem::out << "(" << row << "," << col << "): ";
            (*this)(row, col)->Print();
        }
    }
}

void QuasiMatrixProduct::Lump()
{
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < height; col++) {
            // Set all off diagonal entries to zero
            if (row != col) {
                for (int i = 0; i < height; i++) {
                    *((*this)(row, col))[i] = 0.; 
                }
            // Handle diagonal entries    
            }
            else {
                for (int i = 0; i < height; i++) {
                    *((*this)(row, col))[i] = fabs(*((*this)(row, col))[i]);
                }
                double current = *((*this)(row, col))[0], max = current;
                int maxidx = 0;
                for (int i = 1; i < height; i++) {
                    current = *((*this)(row, col))[i];
                    if (current > max) {
                        *((*this)(row, col))[maxidx] = 0.;
                        max = current;
                        maxidx = i;
                    }
                    else {
                        *((*this)(row, col))[i] = 0.;
                    }
                }
                *((*this)(row,col))[maxidx] = 1.;
            }
        }
    }
}

void QuasiMatrixProduct::TruncateOffDiags()
{
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < height; col++) {
            if (row != col) {
                for (int i = 0; i < height; i++) {
                    *((*this)(row, col))[i] = 0.; 
                }
            }
        }            
    }
}

QuasiMatrixProduct::~QuasiMatrixProduct()
{
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < height; col++) {
            delete (*this)(row, col);
        }
    }
}

// -------------------------------------------------------------------- //
// -------------------------------------------------------------------- //

JacDiagBlock::JacDiagBlock(const Array<int> &offsets_,
    const IRKOperator &IRKOper_, double R00_) 
    : BlockOperator(offsets_), size(1), IRKOper(IRKOper_), offsets(offsets_), 
    dt(0.0), temp_vector(IRKOper_.Height()), R00(R00_)
{
    MFEM_ASSERT(IRKOper.GetExplicitGradientsType() ==
        IRKOperator::ExplicitGradients::APPROXIMATE,
            "JacDiagBlock:: This constructor is for IRKOperator's \
            with ExplicitGradients::APPROXIMATE");
}

JacDiagBlock::JacDiagBlock(const Array<int> &offsets_,
    const IRKOperator &IRKOper_, double R00_, Vector Z00_) 
    : BlockOperator(offsets_), size(1), IRKOper(IRKOper_), offsets(offsets_),
    dt(0.0), temp_vector(IRKOper_.Height()), R00(R00_), Z00(Z00_)
{
    MFEM_ASSERT(IRKOper.GetExplicitGradientsType() ==
        IRKOperator::ExplicitGradients::EXACT,
            "JacDiagBlock:: This constructor is for IRKOperator's \
            with ExplicitGradients::EXACT");
}

JacDiagBlock::JacDiagBlock(const Array<int> &offsets_,
    const IRKOperator &IRKOper_, double R00_, double R01_,
    double R10_, double R11_)
    : BlockOperator(offsets_),
    size(2), IRKOper(IRKOper_), offsets(offsets_), dt(0.0),
    temp_vector(IRKOper_.Height()), R00(R00_), R01(R01_), R10(R10_), R11(R11_)
{
    MFEM_ASSERT(IRKOper.GetExplicitGradientsType() ==
        IRKOperator::ExplicitGradients::APPROXIMATE,
            "JacDiagBlock:: This constructor is for IRKOperator's \
            with ExplicitGradients::APPROXIMATE");
}

JacDiagBlock::JacDiagBlock(const Array<int> &offsets_,
    const IRKOperator &IRKOper_, double R00_, double R01_,
    double R10_, double R11_, 
    Vector Z00_, Vector Z01_, Vector Z10_, Vector Z11_) 
    : BlockOperator(offsets_), size(2), IRKOper(IRKOper_),
    offsets(offsets_), dt(0.0), temp_vector(IRKOper_.Height()),
    R00(R00_), R01(R01_), R10(R10_), R11(R11_), 
    Z00(Z00_), Z01(Z01_), Z10(Z10_), Z11(Z11_) 
{
    MFEM_ASSERT(IRKOper.GetExplicitGradientsType() ==
        IRKOperator::ExplicitGradients::EXACT,
            "JacDiagBlock:: This constructor is for IRKOperator's \
            with ExplicitGradients::EXACT");
}
    
void JacDiagBlock::Mult(const Vector &x, Vector &y) const 
{    
    MFEM_ASSERT(x.Size() == this->Height(), "JacDiagBlock::Mult() incorrect input Vector size");
    MFEM_ASSERT(y.Size() == this->Height(), "JacDiagBlock::Mult() incorrect output Vector size");

    switch (IRKOper.GetExplicitGradientsType()) {
        case IRKOperator::ExplicitGradients::APPROXIMATE:
            // 1x1 operator, y = [R(0)(0)*M - dt*Na']*x
            if (size == 1) {
                IRKOper.ExplicitGradientMult(x, y);
                y *= -dt;
            
            // MASS MATRIX    
            if (IRKOper.isImplicit()) {
                IRKOper.MassMult(x, temp_vector);
                y.Add(R00, temp_vector);
            // NO MASS MATRIX    
            }
            else {
                y.Add(R00, x);
            }

            // 2x2 operator,
            //  y(0) = [R(0,0)*M-dt*Na']*x(0) + [R(0,1)*M]*x(1)
            //  y(1) = [R(1,0)*M]*x(0)        + [R(1,1)*M-dt*Na']*x(1)
            }
            else if (size == 2) {
                // Wrap scalar Vectors with BlockVectors
                x_block.Update(x.GetData(), offsets);
                y_block.Update(y.GetData(), offsets);
                
                // y(0)
                IRKOper.ExplicitGradientMult(x_block.GetBlock(0), y_block.GetBlock(0));
                y_block.GetBlock(0) *= -dt;
                
                // y(1)
                IRKOper.ExplicitGradientMult(x_block.GetBlock(1), y_block.GetBlock(1));
                y_block.GetBlock(1) *= -dt;
                
                // MASS MATRIX
                if (IRKOper.isImplicit()) {
                    // M*x(0) dependence
                    IRKOper.MassMult(x_block.GetBlock(0), temp_vector);
                    y_block.GetBlock(0).Add(R00, temp_vector);
                    y_block.GetBlock(1).Add(R10, temp_vector);
                    // M*x(1) dependence
                    IRKOper.MassMult(x_block.GetBlock(1), temp_vector);
                    y_block.GetBlock(0).Add(R01, temp_vector);
                    y_block.GetBlock(1).Add(R11, temp_vector);
                
                // NO MASS MATRIX    
                }
                else {
                    // x(0) dependence
                    y_block.GetBlock(0).Add(R00, x_block.GetBlock(0));
                    y_block.GetBlock(1).Add(R10, x_block.GetBlock(0));
                    // x(1) dependence
                    y_block.GetBlock(0).Add(R01, x_block.GetBlock(1));
                    y_block.GetBlock(1).Add(R11, x_block.GetBlock(1));
                }
            }
            break;

        case IRKOperator::ExplicitGradients::EXACT:
            /** 1x1 operator, 
                    y = [R(0,0)*M-dt*<z(0,0),{N'}>*x */
            if (size == 1) {
                y = 0.;
                IRKOper.AddExplicitGradientsMult(-dt, Z00, x, y);
                
                // MASS MATRIX    
                if (IRKOper.isImplicit()) {
                    IRKOper.MassMult(x, temp_vector);
                    y.Add(R00, temp_vector);
                // NO MASS MATRIX    
                }
                else {
                    y.Add(R00, x);
                }
            }
            /** 2x2 operator,
                    y(0) = [R(0,0)*M-dt*<Z(0,0),{N'>]*x(0)  + [R(0,1)*M-dt*<Z(0,1),{N'}>]*x(1)
                    y(1) = [R(1,0)*M-dt*<Z(1,0),{N'}>]*x(0) + [R(1,1)*M-dt*<Z(1,1),{N'}>]*x(1) */
            else if (size == 2) {
                // Wrap scalar Vectors with BlockVectors
                x_block.Update(x.GetData(), offsets);
                y_block.Update(y.GetData(), offsets);
                
                // Initialize y to zero
                y_block.GetBlock(0) = 0.;
                y_block.GetBlock(1) = 0.;
                
                // --- Dependence on x(0)
                IRKOper.AddExplicitGradientsMult(-dt, Z00, -dt, Z10, 
                                                x_block.GetBlock(0), 
                                                y_block.GetBlock(0), y_block.GetBlock(1));
                // MASS MATRIX
                if (IRKOper.isImplicit()) {
                    IRKOper.MassMult(x_block.GetBlock(0), temp_vector);
                    y_block.GetBlock(0).Add(R00, temp_vector);
                    y_block.GetBlock(1).Add(R10, temp_vector);
                // NO MASS MATRIX    
                }
                else {
                    y_block.GetBlock(0).Add(R00, x_block.GetBlock(0));
                    y_block.GetBlock(1).Add(R10, x_block.GetBlock(0));
                }
                
                // --- Dependence on x(1)
                IRKOper.AddExplicitGradientsMult(-dt, Z01, -dt, Z11, 
                                                x_block.GetBlock(1), 
                                                y_block.GetBlock(0), y_block.GetBlock(1));
                // MASS MATRIX
                if (IRKOper.isImplicit()) {
                    IRKOper.MassMult(x_block.GetBlock(1), temp_vector);
                    y_block.GetBlock(0).Add(R01, temp_vector);
                    y_block.GetBlock(1).Add(R11, temp_vector);
                // NO MASS MATRIX    
                }
                else {
                    y_block.GetBlock(0).Add(R01, x_block.GetBlock(1));
                    y_block.GetBlock(1).Add(R11, x_block.GetBlock(1));
                }
            }
            break;
    }
}     

// -------------------------------------------------------------------- //
// -------------------------------------------------------------------- //

JacDiagBlockPrec::JacDiagBlockPrec(const JacDiagBlock &BlockOper_, bool identity_) 
    : Solver(BlockOper_.Height()), BlockOper(BlockOper_), 
    identity(identity_) 
{

}
    
JacDiagBlockPrec::JacDiagBlockPrec(const JacDiagBlock &BlockOper_, double R10_, 
    int prec00_idx_, int prec11_idx_, bool identity_) 
    : Solver(BlockOper_.Height()), BlockOper(BlockOper_), 
        identity(identity_), temp_vector(BlockOper_.Offsets()[1]),
        R10(R10_), prec00_idx(prec00_idx_), prec11_idx(prec11_idx_)  
{
    MFEM_ASSERT(BlockOper.IRKOper.GetExplicitGradientsType() ==
        IRKOperator::ExplicitGradients::APPROXIMATE,
            "JacDiagBlockPrec:: This constructor is for IRKOperator's \
            with ExplicitGradients::APPROXIMATE");
}

JacDiagBlockPrec::JacDiagBlockPrec(const JacDiagBlock &BlockOper_, double R10_, Vector Y10_, 
    int prec00_idx_, int prec11_idx_, bool identity_) 
    : Solver(BlockOper_.Height()), BlockOper(BlockOper_), 
        identity(identity_), temp_vector(BlockOper_.Offsets()[1]),
        R10(R10_), Y10(Y10_), prec00_idx(prec00_idx_), prec11_idx(prec11_idx_) 
{
    MFEM_ASSERT(BlockOper.IRKOper.GetExplicitGradientsType() ==
        IRKOperator::ExplicitGradients::EXACT,
            "JacDiagBlockPrec:: This constructor is for IRKOperator's \
            with ExplicitGradients::EXACT");
}

void JacDiagBlockPrec::Mult(const Vector &x_vector, Vector &y_vector) const {
    // Use an identity preconditioner
    if (identity) {
        y_vector = x_vector;
        
    // Use a proper preconditioner    
    }
    else {
        // 1x1 system
        if (BlockOper.Size() == 1) {
            BlockOper.IRKOper.ImplicitPrec(x_vector, y_vector);
        }
        
        /* 2x2 system uses 2x2 block lower triangular preconditioner,
            [A 0][y0] = x0  =>  y0 = A^{-1}*x0
            [C D][y1] = x1  =>  y1 = D^{-1}*(x1 - C*y0) */
        else if (BlockOper.Size() == 2) {
            // Wrap scalar Vectors with BlockVectors
            x_block.Update(x_vector.GetData(), BlockOper.Offsets());
            y_block.Update(y_vector.GetData(), BlockOper.Offsets());
            
            
            // Which system is solved depends on IRKOper::ExplicitGradients
            switch (BlockOper.IRKOper.GetExplicitGradientsType()) {
                
                // C == R(1,0)*M
                case IRKOperator::ExplicitGradients::APPROXIMATE:
                    // Approximately invert (0,0) block 
                    BlockOper.IRKOper.ImplicitPrec(prec00_idx, x_block.GetBlock(0), y_block.GetBlock(0));
                    
                    // Form RHS of next system, temp <- x(1) - C*y(0)
                    // MASS MATRIX
                    if (BlockOper.IRKOper.isImplicit()) {
                        BlockOper.IRKOper.MassMult(y_block.GetBlock(0), temp_vector);
                        temp_vector *= -R10;
                        temp_vector += x_block.GetBlock(1);
                    // NO MASS MATRIX    
                    }
                    else {
                        add(x_block.GetBlock(1), -R10, y_block.GetBlock(0), temp_vector); 
                    }
                    
                    // Approximately invert (1,1) block
                    BlockOper.IRKOper.ImplicitPrec(prec11_idx, temp_vector, y_block.GetBlock(1));
                    break;
                
                
                // C == R(1,0)*M-dt*<Y(1,0),{N'}> 
                case IRKOperator::ExplicitGradients::EXACT:
                    // Approximately invert (0,0) block 
                    BlockOper.IRKOper.ImplicitPrec(prec00_idx, x_block.GetBlock(0), y_block.GetBlock(0));
                    
                    // Form RHS of next system, temp <- x(1) - C*y(0)
                    temp_vector = x_block.GetBlock(1);
                    // MASS MATRIX
                    if (BlockOper.IRKOper.isImplicit()) {
                        BlockOper.IRKOper.MassMult(y_block.GetBlock(0), y_block.GetBlock(1));
                        temp_vector.Add(-R10, y_block.GetBlock(1));
                    // NO MASS MATRIX    
                    }
                    else {
                        temp_vector.Add(-R10, y_block.GetBlock(0));
                    }    
                    BlockOper.IRKOper.AddExplicitGradientsMult(-BlockOper.GetTimeStep(), Y10, 
                                                                y_block.GetBlock(0), temp_vector);
                    
                    // Approximately invert (1,1) block
                    BlockOper.IRKOper.ImplicitPrec(prec11_idx, temp_vector, y_block.GetBlock(1));
                    break;
            }
        }
    }
}

// -------------------------------------------------------------------- //
// -------------------------------------------------------------------- //

TriJacSolver::TriJacSolver(IRKStageOper &StageOper_, int jac_update_rate_, int gamma_idx_,
    const KrylovParams &solver_params1, const KrylovParams &solver_params2,
    const QuasiMatrixProduct * Z_solver_, const QuasiMatrixProduct * Z_prec_) 
    : Solver(StageOper_.Height()),
    StageOper(StageOper_),
    jac_update_rate(jac_update_rate_), gamma_idx(gamma_idx_),
    printlevel(solver_params1.printlevel),
    offsets(StageOper_.RowOffsets()),
    x_block(StageOper_.RowOffsets()), b_block(StageOper_.RowOffsets()), 
    b_block_temp(StageOper_.RowOffsets()), x_block_temp(StageOper_.RowOffsets()), 
    temp_vector1(StageOper_.RowOffsets()[1]), temp_vector2(StageOper_.RowOffsets()[1]),
    krylov_solver1(NULL), krylov_solver2(NULL), multiple_krylov(false),
    Z_solver(Z_solver_), Z_prec(Z_prec_)
{
    kronecker_form = (StageOper.IRKOper->GetExplicitGradientsType() ==
        IRKOperator::ExplicitGradients::APPROXIMATE);

    // Ensure that valid Z_solver and Z_prec provided for non Kronecker Jacobian
    if (!kronecker_form) {
        MFEM_ASSERT((Z_solver) && (Z_prec), "TriJacSolver:: IRKOperator using \
        exact gradients requires non NULL sparsity patterns Z_solver and Z_prec");
    }

    // Create offset arrays for 1x1 and 2x2 operators
    offsets.GetSubArray(0, 2, offsets_1);
    if (offsets.Size() > 2) offsets.GetSubArray(0, 3, offsets_2);

    // Create operators describing diagonal blocks
    bool size1_solves = false; // Do we solve any 1x1 systems?
    bool size2_solves = false; // Do we solve any 2x2 systems?
    double R00, R01, R10, R11; // Elements from diagonal block of R0
    bool identity = false;     // Use identity preconditioners as test that preconditioners are doing something
    
    int s_eff = StageOper.Butcher.s_eff;
    Array<int> size = StageOper.Butcher.R0_block_sizes;
    /*  Initialize operators describing diagonal blocks and their 
        preconditioners, going from top left to bottom right. */
    DiagBlock.SetSize(s_eff);
    DiagBlockPrec.SetSize(s_eff);
    int row = 0;
    for (int block = 0; block < s_eff; block++) {
        
        // 1x1 diagonal block spanning row=row,col=row
        if (size[block] == 1) {
            size1_solves = true;
            R00 = StageOper.Butcher.R0(row,row);
            
            // Form of operator depends on IRKOperator::ExplicitGradients
            if (kronecker_form) {
                DiagBlock[block] = new JacDiagBlock(offsets_1, *(StageOper.IRKOper), R00);    
            }
            else {
                DiagBlock[block] = new JacDiagBlock(offsets_1, *(StageOper.IRKOper), R00, *(*Z_solver)(row,row));    
            }
            DiagBlockPrec[block] = new JacDiagBlockPrec(*(DiagBlock[block]), identity);
            
        
        // 2x2 diagonal block spanning rows=(row,row+1),cols=(row,row+1)
        }
        else if (size[block] == 2) {
            size2_solves = true;                
            R00 = StageOper.Butcher.R0(row,row);
            R01 = StageOper.Butcher.R0(row,row+1);
            R10 = StageOper.Butcher.R0(row+1,row);
            R11 = StageOper.Butcher.R0(row+1,row+1);
            
            // Form of operator and preconditioner depends on IRKOperator::ExplicitGradients
            if (kronecker_form) {
                DiagBlock[block] = new JacDiagBlock(offsets_2, *(StageOper.IRKOper), 
                                                R00, R01, R10, R11);
                                                
                // Diagonal blocks in preconditioner are the same
                if (gamma_idx == 0) {
                    DiagBlockPrec[block] = new JacDiagBlockPrec(*(DiagBlock[block]),
                                                    R10, 
                                                    row,  // Precondition (row,row) block with IRKOper.ImplicitPrec(row,.,.)
                                                    row,  // Precondition (row+1,row+1) block with IRKOper.ImplicitPrec(row,.,.)
                                                    identity);                                                                
                // Diagonal blocks in preconditioner are different
                }
                else {
                    DiagBlockPrec[block] = new JacDiagBlockPrec(*(DiagBlock[block]),
                                                    R10, 
                                                    row,  // Precondition (row,row) block with IRKOper.ImplicitPrec(row,.,.)
                                                    row+1,// Precondition (row+1,row+1) block with IRKOper.ImplicitPrec(row+1,.,.)
                                                    identity);                                
                }
                
            }
            else {
                DiagBlock[block] = new JacDiagBlock(offsets_2, *(StageOper.IRKOper), 
                                                R00, R01, R10, R11, 
                                                *(*Z_solver)(row,row), 
                                                *(*Z_solver)(row,row+1), 
                                                *(*Z_solver)(row+1,row), 
                                                *(*Z_solver)(row+1,row+1));
                DiagBlockPrec[block] = new JacDiagBlockPrec(*(DiagBlock[block]),
                                                R10, 
                                                *(*Z_prec)(row+1,row), 
                                                row,  // Precondition (row,row) block with IRKOper.ImplicitPrec(row,.,.)
                                                row+1,// Precondition (row+1,row+1) block with IRKOper.ImplicitPrec(row+1,.,.)
                                                identity);
            }
        }
        else {
            mfem_error("TriJacSolver:: R0 block sizes must be 1 or 2");
        }
        
        // Increment row counter by current block size
        row += size[block];
    }
    
    // Set up Krylov solver 
    GetKrylovSolver(krylov_solver1, solver_params1);
    krylov_solver2 = krylov_solver1; // By default, 2x2 systems solved with krylov_solver1.
    
    /*  Setup different solver for 2x2 blocks if needed (solving both 1x1 and 
        2x2 systems AND references to solver parameters are not identical) */
    if ((size1_solves && size2_solves) && (&solver_params1 != &solver_params2)) {
        MFEM_ASSERT(solver_params2.solver == KrylovMethod::GMRES, 
                        "TriJacSolver:: 2x2 systems must use GMRES.\n");
        GetKrylovSolver(krylov_solver2, solver_params2);
        multiple_krylov = true;
    }
    
    krylov_iters.resize(s_eff, 0);
}

TriJacSolver::~TriJacSolver()
{
    for (int i = 0; i < DiagBlock.Size(); i++) {
        delete DiagBlockPrec[i];
        delete DiagBlock[i];
    }
    delete krylov_solver1;
    if (multiple_krylov) delete krylov_solver2;
}

void TriJacSolver::ResetNumIterations()
{ 
    for (int i = 0; i < krylov_iters.size(); i++) krylov_iters[i] = 0; 
}

void TriJacSolver::SetOperator(const Operator &op)
{ 
    // Update gradient(s) if: not linearly implicit, and either first Newton iteration,
    // OR current iteration is a multiple of update rate  
    if (!StageOper.IRKOper->IsLinearlyImplicit() && (
            StageOper.GetGradientCalls() == 1 || 
            (jac_update_rate > 0 && 
                (StageOper.GetGradientCalls()+1) % jac_update_rate == 0) ) )
    {
        // IRK methods, where *(StageOper.u) is not null
        if (StageOper.u) {
            if (kronecker_form) {
                // Set approximate gradient Na' 
                StageOper.IRKOper->SetExplicitGradient(*(StageOper.u), StageOper.GetTimeStep(), 
                                        StageOper.GetCurrentIterate(), StageOper.Butcher.c0);
            }
            else {
                // Set exact gradients {N'} 
                StageOper.IRKOper->SetExplicitGradients(*(StageOper.u), StageOper.GetTimeStep(), 
                                        StageOper.GetCurrentIterate(), StageOper.Butcher.c0);
            }
        }
        // PolyIMEX methods, *(StageOper.u) is null
        else {
            if (kronecker_form) {
                // Set approximate gradient Na' 
                StageOper.IRKOper->SetExplicitGradient(StageOper.GetTimeStep(), 
                                        StageOper.GetCurrentIterate(),
                                        StageOper.Butcher.z0);
            }
            else {
                // Set exact gradients {N'} 
                StageOper.IRKOper->SetExplicitGradients(StageOper.GetTimeStep(), 
                                        StageOper.GetCurrentIterate(),
                                        StageOper.Butcher.z0);
            }
        }
    }
}

void TriJacSolver::Mult(const Vector &b_vector, Vector &x_vector) const
{
    // Wrap scalar Vectors into BlockVectors
    b_block.Update(b_vector.GetData(), offsets);
    x_block.Update(x_vector.GetData(), offsets);
 
    // Transform initial guess and RHS 
    KronTransformTranspose(StageOper.Butcher.Q0, x_block, x_block_temp);
    KronTransformTranspose(StageOper.Butcher.Q0, b_block, b_block_temp);
    
    // Solve \tilde{J}*x_block_temp=b_block_temp, 
    BlockBackwardSubstitution(b_block_temp, x_block_temp);

    // Transform to get original x
    KronTransform(StageOper.Butcher.Q0, x_block_temp, x_block); 
}

void TriJacSolver::GetKrylovSolver(IterativeSolver * &solver,
    const KrylovParams &params) const 
{
    switch (params.solver) {
        case KrylovMethod::FP:
            solver = new SLISolver(StageOper.IRKOper->GetComm());
            break;
        case KrylovMethod::CG:
            solver = new CGSolver(StageOper.IRKOper->GetComm());
            break;
        case KrylovMethod::MINRES:
            solver = new MINRESSolver(StageOper.IRKOper->GetComm());
            break;
        case KrylovMethod::GMRES:
            solver = new GMRESSolver(StageOper.IRKOper->GetComm());
            static_cast<GMRESSolver*>(solver)->SetKDim(params.kdim);
            break;
        case KrylovMethod::BICGSTAB:
            solver = new MINRESSolver(StageOper.IRKOper->GetComm());
            break;    
        case KrylovMethod::FGMRES:
            solver = new FGMRESSolver(StageOper.IRKOper->GetComm());
            static_cast<FGMRESSolver*>(solver)->SetKDim(params.kdim);
            break;
        default:
            mfem_error("Invalid Krylov solve type.\n");   
    }
    
    solver->iterative_mode = params.iterative_mode;
    solver->SetAbsTol(params.abstol);
    solver->SetRelTol(params.reltol);
    solver->SetMaxIter(params.maxiter);
    solver->SetPrintLevel(params.printlevel);
}

void TriJacSolver::BlockBackwardSubstitution(BlockVector &z_block,
    BlockVector &y_block) const
{
    if (printlevel > 0) mfem::out << "  ---Backward solve---" << '\n';
    
    // Short hands
    int s = StageOper.Butcher.s; 
    int s_eff = StageOper.Butcher.s_eff; 
    const DenseMatrix &R = StageOper.Butcher.R0;
    Array<int> size = StageOper.Butcher.R0_block_sizes;
    bool krylov_converged;
    
    double dt = StageOper.GetTimeStep();
    
    /** Backward substitution: Invert diagonal blocks, which are:
        -1x1 systems for y(row), or 
        -2x2 systems for (y(row),y(row+1)) */
    int row = s; 
    for (int diagBlock = s_eff-1; diagBlock >= 0; diagBlock--)
    {
        int solve = s_eff - diagBlock; 
        if (printlevel > 0) mfem::out << "    Block solve " << solve << " of " << s_eff;
        
        // Decrement row counter by current block size.
        row -= size[diagBlock];
        
        // Update parameters for diagonal blocks
        DiagBlock[diagBlock]->SetTimeStep(StageOper.GetTimeStep());
        
        // Compute constant gamma used to precondition Schur complement of 2x2 block
        double gamma = 0.;
        if (size[diagBlock] == 2) {
            double eta = R(row,row), beta = std::sqrt(-R(row,row+1)*R(row+1,row));
            if (gamma_idx == 0) {
                gamma = eta;
            }
            else if (gamma_idx == 1) {
                gamma = eta + beta*beta/eta;
            }
            else {
                mfem_error("gamma must be 0, 1");
            }
        }
        
        // Assemble preconditioner(s) for diag block
        if (kronecker_form) {
            // Preconditioner for R(row,row)*M-dt*Na' 
            StageOper.IRKOper->SetPreconditioner(row, dt, R(row,row), size[diagBlock]);
            
            /* Inverting 2x2 block: Assemble a 2nd preconditioner for 
                gamma*M-dt*Na' if gamma is not eta */
            if (size[diagBlock] == 2 && gamma_idx != 0) {
                StageOper.IRKOper->SetPreconditioner(row+1, dt, gamma, size[diagBlock]);
            }
            
        }
        else {
            // Preconditioner for R(row,row)*M-dt*<Z_prec(row,row),{N'}> 
            StageOper.IRKOper->SetPreconditioner(row, dt, R(row,row), *(*Z_prec)(row,row));

            /* Inverting 2x2 block: Assemble a 2nd preconditioner for 
                gamma*M-dt*<Z_prec(row+1,row+1),{N'}> */
            if (size[diagBlock] == 2) {
                StageOper.IRKOper->SetPreconditioner(row+1, dt, gamma, *(*Z_prec)(row+1,row+1));
            }
        }    
        
        // Invert 1x1 diagonal block
        if (size[diagBlock] == 1) 
        {
            if (printlevel > 0) {
                mfem::out << ": 1x1 block  -->  ";
                if (printlevel != 2) mfem::out << '\n';
            }
            // --- Form RHS vector (this overrides z_block(row)) --- //
            // Subtract out known information from LHS of equations
            if (row+1 < s) {
                /// R0 component
                // MASS MATRIX
                if (StageOper.IRKOper->isImplicit()) {
                    temp_vector1.Set(-R(row,row+1), y_block.GetBlock(row+1));
                    for (int j = row+2; j < s; j++) {
                        temp_vector1.Add(-R(row,j), y_block.GetBlock(j));
                    }
                    StageOper.IRKOper->MassMult(temp_vector1, temp_vector2);
                    z_block.GetBlock(row) += temp_vector2; // Add to existing RHS
                    
                // NO MASS MATRIX    
                }
                else {
                    for (int j = row+1; j < s; j++) {
                        z_block.GetBlock(row).Add(-R(row,j), y_block.GetBlock(j));
                    }
                }
                /// {N'} components (only appear for non-Kronecker product form)
                if (!kronecker_form) {
                    for (int j = row+1; j < s; j++) {
                        StageOper.IRKOper->AddExplicitGradientsMult(
                                                dt, *(*Z_solver)(row,j),
                                                y_block.GetBlock(j),
                                                z_block.GetBlock(row));
                    }                        
                }
            }
            
            // --- Solve 1x1 system --- 
            // Pass preconditioner for diagonal block to Krylov solver
            krylov_solver1->SetPreconditioner(*DiagBlockPrec[diagBlock]);
            // Pass diagonal block to Krylov solver
            krylov_solver1->SetOperator(*DiagBlock[diagBlock]);
            // Solve
            krylov_solver1->Mult(z_block.GetBlock(row), y_block.GetBlock(row));
            krylov_converged = krylov_solver1->GetConverged();
            krylov_iters[diagBlock] += krylov_solver1->GetNumIterations();
        } 
        // Invert 2x2 diagonal block
        else if (size[diagBlock] == 2) 
        {
            if (printlevel > 0) {
                mfem::out << ": 2x2 block  -->  ";
                if (printlevel != 2) mfem::out << '\n';
            }
    
            // --- Form RHS vector (this overrides z_block(row),z_block(row+1)) --- //
            // Point z_2block to the appropriate data from z_block 
            // (note data arrays for blocks are stored contiguously)
            z_2block.Update(z_block.GetBlock(row).GetData(), offsets_2);
            
            // Subtract out known information from LHS of equations
            if (row+2 < s) {
                /// R0 component
                // MASS MATRIX
                if (StageOper.IRKOper->isImplicit()) {
                    // First component
                    temp_vector1.Set(-R(row,row+2), y_block.GetBlock(row+2));
                    for (int j = row+3; j < s; j++) {
                        temp_vector1.Add(-R(row,j), y_block.GetBlock(j));
                    }
                    StageOper.IRKOper->MassMult(temp_vector1, temp_vector2);
                    z_2block.GetBlock(0) += temp_vector2; // Add to existing RHS
                    // Second component
                    temp_vector1.Set(-R(row+1,row+2), y_block.GetBlock(row+2)); 
                    for (int j = row+3; j < s; j++) {
                        temp_vector1.Add(-R(row+1,j), y_block.GetBlock(j)); 
                    }
                    StageOper.IRKOper->MassMult(temp_vector1, temp_vector2);
                    z_2block.GetBlock(1) += temp_vector2; // Add to existing RHS
                
                // NO MASS MATRIX    
                }
                else {
                    for (int j = row+2; j < s; j++) {
                        z_2block.GetBlock(0).Add(-R(row,j), y_block.GetBlock(j)); // First component
                        z_2block.GetBlock(1).Add(-R(row+1,j), y_block.GetBlock(j)); // Second component
                    }
                }
                
                /// {N'} components (only appear for non-Kronecker product form)
                if (!kronecker_form) {
                    for (int j = row+2; j < s; j++) {                
                        StageOper.IRKOper->AddExplicitGradientsMult(
                                                dt, *(*Z_solver)(row,j), 
                                                dt, *(*Z_solver)(row+1,j), 
                                                y_block.GetBlock(j), 
                                                z_2block.GetBlock(0), z_2block.GetBlock(1));
                    }
                }
            }
            
            // Point y_2block to data array of solution vector
            y_2block.Update(y_block.GetBlock(row).GetData(), offsets_2);
            
            // --- Solve 2x2 system --- 
            // Pass preconditioner for diagonal block to Krylov solver
            krylov_solver2->SetPreconditioner(*DiagBlockPrec[diagBlock]);
            // Pass diagonal block to Krylov solver
            krylov_solver2->SetOperator(*DiagBlock[diagBlock]);
            // Solve
            krylov_solver2->Mult(z_2block, y_2block);
            krylov_converged = krylov_solver2->GetConverged();    
            krylov_iters[diagBlock] += krylov_solver1->GetNumIterations();
        }
        
        // Check convergence 
        if (!krylov_converged) {
            string msg = "KronJacSolver::BlockBackwardSubstitution() Krylov solver at t=" 
                            + to_string(StageOper.IRKOper->GetTime()) 
                            + " not converged [system " + to_string(solve) 
                            + "/" + to_string(s_eff) 
                            + ", size=" + to_string(size[diagBlock]) + ")]\n";
            mfem_error(msg.c_str());
        }
    }
}

// -------------------------------------------------------------------- //
// -------------------------------------------------------------------- //

/// Set dimensions of data structures 
void RKData::SizeData()
{
    // Basic `Butcher Tableau` data
    invA0.SetSize(s);
    if (is_imex) {
        A0.SetSize(s+1);
        // A0_it.SetSize(s+1);      // *--- IF A0_it != A0 ---*
        expA0.SetSize(s+1);
        expA0_it.SetSize(s+1);
        z0.SetSize(s+1);
    }
    else {
        A0.SetSize(s);
        c0.SetSize(s);
        b0.SetSize(s);
        d0.SetSize(s);
    }
  
    // NOTE:
    //     s := 2*n(cc_eig_pairs) + n(r_eigs)
    // s_eff := n(cc_eig_pairs) + n(r_eigs)
    // ==>  n(cc_eig_pairs) = s - s_eff
    // ==>  n(r_eigs) = 2*s_eff - s    
    Q0.SetSize(s);  
    R0.SetSize(s);  
    R0_block_sizes.SetSize(s_eff);
}


/** Set dummy 2x2 RK data that has the specified value of beta_on_eta == beta/eta
    and real part of eigenvalue equal to eta_ */
void RKData::SetDummyData(double beta_on_eta, double eta_)
{
    s = 2;      // 2 stages
    s_eff = 1;  // Have one 2x2 system
    
    SizeData();
    /* --- R block sizes --- */
    R0_block_sizes[0] = 2;
    
    double phi = 1.0; // Some non-zero constant...
    double a = eta_;
    double b = phi;
    double c = -beta_on_eta*eta_*beta_on_eta*eta_/phi;
    double d = eta_;
    
    /* --- inv(A) --- */
    invA0(0, 0) = a;
    invA0(0, 1) = b;
    invA0(1, 0) = c;
    invA0(1, 1) = d;
    
    /* --- A --- */
    double det = a*d - b*d;
    A0(0, 0) = d/det;
    A0(0, 1) = -b/det;
    A0(1, 0) = -c/det;
    A0(1, 1) = a/det;
    
    /* --- Q --- */
    // choose Q as identity, so inv(A) is equal to R.
    Q0(0, 0) = 1.0;
    Q0(0, 1) = 0.0;
    Q0(1, 0) = 0.0;
    Q0(1, 1) = 1.0;
    /* --- R --- */
    // Set R = inv(A0)
    R0(0, 0) = a;
    R0(0, 1) = b;
    R0(1, 0) = c;
    R0(1, 1) = d;
    
    /* --- b --- */
    // These are just non-zero dummy values.
    b0(0) = 0.5;
    b0(1) = 0.5;
    /* --- c --- */
    // c is the row sums of A
    c0(0) = A0(0, 0) + A0(0, 1);
    c0(1) = A0(1, 0) + A0(1, 1);
    /* --- d --- */
    // d = inv(A0)^T * b
    d0(0) = invA0(0, 0)*b0(0) + invA0(1, 0)*b0(1);
    d0(1) = invA0(0, 1)*b0(0) + invA0(1, 1)*b0(1);;
}


/// Set data required by solvers 
void RKData::SetData() {
    switch(ID) {
        // 2-stage 3rd-order A-stable SDIRK
        case Type::ASDIRK3:
            is_imex = false;
            s = 2;
            s_eff = 2;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.788675134594813;
            A0(0, 1) = +0.000000000000000;
            A0(1, 0) = -0.577350269189626;
            A0(1, 1) = +0.788675134594813;
            /* --- inv(A) --- */
            invA0(0, 0) = +1.267949192431123;
            invA0(0, 1) = -0.000000000000000;
            invA0(1, 0) = +0.928203230275509;
            invA0(1, 1) = +1.267949192431123;
            /* --- b --- */
            b0(0) = +0.500000000000000;
            b0(1) = +0.500000000000000;
            /* --- c --- */
            c0(0) = +0.788675134594813;
            c0(1) = +0.211324865405187;
            /* --- d --- */
            d0(0) = +1.098076211353316;
            d0(1) = +0.633974596215561;
            /* --- Q --- */
            Q0(0, 0) = +0.000000000000000;
            Q0(0, 1) = +1.000000000000000;
            Q0(1, 0) = -1.000000000000000;
            Q0(1, 1) = +0.000000000000000;
            /* --- R --- */
            R0(0, 0) = +1.267949192431123;
            R0(0, 1) = -0.928203230275509;
            R0(1, 0) = +0.000000000000000;
            R0(1, 1) = +1.267949192431123;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 1;
            R0_block_sizes[1] = 1;  
            
        // 3-stage 4th-order A-stable SDIRK
        case Type::ASDIRK4:
            is_imex = false;
            s = 3;
            s_eff = 3;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +1.068579021301629;
            A0(0, 1) = +0.000000000000000;
            A0(0, 2) = +0.000000000000000;
            A0(1, 0) = -0.568579021301629;
            A0(1, 1) = +1.068579021301629;
            A0(1, 2) = +0.000000000000000;
            A0(2, 0) = +2.137158042603258;
            A0(2, 1) = -3.274316085206515;
            A0(2, 2) = +1.068579021301629;
            /* --- inv(A) --- */
            invA0(0, 0) = +0.935822227524088;
            invA0(0, 1) = -0.000000000000000;
            invA0(0, 2) = +0.000000000000000;
            invA0(1, 0) = +0.497940606760015;
            invA0(1, 1) = +0.935822227524088;
            invA0(1, 2) = +0.000000000000000;
            invA0(2, 0) = -0.345865915800969;
            invA0(2, 1) = +2.867525668568206;
            invA0(2, 2) = +0.935822227524088;
            /* --- b --- */
            b0(0) = +0.128886400515720;
            b0(1) = +0.742227198968559;
            b0(2) = +0.128886400515720;
            /* --- c --- */
            c0(0) = +1.068579021301629;
            c0(1) = +0.500000000000000;
            c0(2) = -0.068579021301629;
            /* --- d --- */
            d0(0) = +0.445622407287714;
            d0(1) = +1.064177772475912;
            d0(2) = +0.120614758428183;
            /* --- Q --- */
            Q0(0, 0) = +0.000000000000000;
            Q0(0, 1) = +0.000000000000000;
            Q0(0, 2) = -1.000000000000000;
            Q0(1, 0) = +0.000000000000000;
            Q0(1, 1) = +1.000000000000000;
            Q0(1, 2) = +0.000000000000000;
            Q0(2, 0) = -1.000000000000000;
            Q0(2, 1) = +0.000000000000000;
            Q0(2, 2) = +0.000000000000000;
            /* --- R --- */
            R0(0, 0) = +0.935822227524088;
            R0(0, 1) = -2.867525668568206;
            R0(0, 2) = -0.345865915800969;
            R0(1, 0) = +0.000000000000000;
            R0(1, 1) = +0.935822227524088;
            R0(1, 2) = -0.497940606760015;
            R0(2, 0) = +0.000000000000000;
            R0(2, 1) = +0.000000000000000;
            R0(2, 2) = +0.935822227524088;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 1;
            R0_block_sizes[1] = 1;
            R0_block_sizes[2] = 1;
            break;

        // 1-stage 1st-order L-stable SDIRK
        case Type::LSDIRK1:
            is_imex = false;
            s = 1;
            s_eff = 1;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +1.000000000000000;
            /* --- inv(A) --- */
            invA0(0, 0) = +1.000000000000000;
            /* --- b --- */
            b0(0) = +1.000000000000000;
            /* --- c --- */
            c0(0) = +1.000000000000000;
            /* --- d --- */
            d0(0) = +1.000000000000000;
            /* --- Q --- */
            Q0(0, 0) = -1.000000000000000;
            /* --- R --- */
            R0(0, 0) = +1.000000000000000;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 1;
            break;

        // 2-stage 2nd-order L-stable SDIRK
        case Type::LSDIRK2:
            is_imex = false;
            s = 2;
            s_eff = 2;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.292893218813453;
            A0(0, 1) = +0.000000000000000;
            A0(1, 0) = +0.414213562373095;
            A0(1, 1) = +0.292893218813453;
            /* --- inv(A) --- */
            invA0(0, 0) = +3.414213562373094;
            invA0(0, 1) = +0.000000000000000;
            invA0(1, 0) = -4.828427124746186;
            invA0(1, 1) = +3.414213562373094;
            /* --- b --- */
            b0(0) = +0.500000000000000;
            b0(1) = +0.500000000000000;
            /* --- c --- */
            c0(0) = +0.292893218813453;
            c0(1) = +0.707106781186547;
            /* --- d --- */
            d0(0) = -0.707106781186546;
            d0(1) = +1.707106781186547;
            /* --- Q --- */
            Q0(0, 0) = +0.000000000000000;
            Q0(0, 1) = +1.000000000000000;
            Q0(1, 0) = -1.000000000000000;
            Q0(1, 1) = +0.000000000000000;
            /* --- R --- */
            R0(0, 0) = +3.414213562373094;
            R0(0, 1) = +4.828427124746187;
            R0(1, 0) = +0.000000000000000;
            R0(1, 1) = +3.414213562373094;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 1;
            R0_block_sizes[1] = 1;
            break;

        // 3-stage 3rd-order L-stable SDIRK
        case Type::LSDIRK3:
            is_imex = false;
            s = 3;
            s_eff = 3;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.435866521508459;
            A0(0, 1) = +0.000000000000000;
            A0(0, 2) = +0.000000000000000;
            A0(1, 0) = +0.282066739245771;
            A0(1, 1) = +0.435866521508459;
            A0(1, 2) = +0.000000000000000;
            A0(2, 0) = +1.208496649176010;
            A0(2, 1) = -0.644363170684469;
            A0(2, 2) = +0.435866521508459;
            /* --- inv(A) --- */
            invA0(0, 0) = +2.294280360279042;
            invA0(0, 1) = -0.000000000000000;
            invA0(0, 2) = -0.000000000000000;
            invA0(1, 0) = -1.484721005641544;
            invA0(1, 1) = +2.294280360279042;
            invA0(1, 2) = +0.000000000000000;
            invA0(2, 0) = -8.556127801552645;
            invA0(2, 1) = +3.391748836942547;
            invA0(2, 2) = +2.294280360279042;
            /* --- b --- */
            b0(0) = +1.208496649176010;
            b0(1) = -0.644363170684469;
            b0(2) = +0.435866521508459;
            /* --- c --- */
            c0(0) = +0.435866521508459;
            c0(1) = +0.717933260754229;
            c0(2) = +1.000000000000000;
            /* --- d --- */
            d0(0) = +0.000000000000000;
            d0(1) = +0.000000000000000;
            d0(2) = +1.000000000000000;
            /* --- Q --- */
            Q0(0, 0) = +0.000000000000000;
            Q0(0, 1) = +0.000000000000000;
            Q0(0, 2) = -1.000000000000000;
            Q0(1, 0) = +0.000000000000000;
            Q0(1, 1) = +1.000000000000000;
            Q0(1, 2) = +0.000000000000000;
            Q0(2, 0) = -1.000000000000000;
            Q0(2, 1) = +0.000000000000000;
            Q0(2, 2) = +0.000000000000000;
            /* --- R --- */
            R0(0, 0) = +2.294280360279042;
            R0(0, 1) = -3.391748836942547;
            R0(0, 2) = -8.556127801552645;
            R0(1, 0) = +0.000000000000000;
            R0(1, 1) = +2.294280360279042;
            R0(1, 2) = +1.484721005641544;
            R0(2, 0) = +0.000000000000000;
            R0(2, 1) = +0.000000000000000;
            R0(2, 2) = +2.294280360279042;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 1;
            R0_block_sizes[1] = 1;
            R0_block_sizes[2] = 1;
            break;

        // 5-stage 4th-order L-stable SDIRK
        case Type::LSDIRK4:
            is_imex = false;
            s = 5;
            s_eff = 5;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.250000000000000;
            A0(0, 1) = +0.000000000000000;
            A0(0, 2) = +0.000000000000000;
            A0(0, 3) = +0.000000000000000;
            A0(0, 4) = +0.000000000000000;
            A0(1, 0) = +0.500000000000000;
            A0(1, 1) = +0.250000000000000;
            A0(1, 2) = +0.000000000000000;
            A0(1, 3) = +0.000000000000000;
            A0(1, 4) = +0.000000000000000;
            A0(2, 0) = +0.340000000000000;
            A0(2, 1) = -0.040000000000000;
            A0(2, 2) = +0.250000000000000;
            A0(2, 3) = +0.000000000000000;
            A0(2, 4) = +0.000000000000000;
            A0(3, 0) = +0.272794117647059;
            A0(3, 1) = -0.050367647058824;
            A0(3, 2) = +0.027573529411765;
            A0(3, 3) = +0.250000000000000;
            A0(3, 4) = +0.000000000000000;
            A0(4, 0) = +1.041666666666667;
            A0(4, 1) = -1.020833333333333;
            A0(4, 2) = +7.812500000000000;
            A0(4, 3) = -7.083333333333333;
            A0(4, 4) = +0.250000000000000;
            /* --- inv(A) --- */
            invA0(0, 0) = +4.000000000000000;
            invA0(0, 1) = +0.000000000000000;
            invA0(0, 2) = +0.000000000000000;
            invA0(0, 3) = +0.000000000000000;
            invA0(0, 4) = +0.000000000000000;
            invA0(1, 0) = -7.999999999999995;
            invA0(1, 1) = +3.999999999999998;
            invA0(1, 2) = +0.000000000000001;
            invA0(1, 3) = -0.000000000000003;
            invA0(1, 4) = +0.000000000000000;
            invA0(2, 0) = -6.719999999999998;
            invA0(2, 1) = +0.639999999999999;
            invA0(2, 2) = +4.000000000000001;
            invA0(2, 3) = -0.000000000000002;
            invA0(2, 4) = -0.000000000000000;
            invA0(3, 0) = -5.235294117647047;
            invA0(3, 1) = +0.735294117647057;
            invA0(3, 2) = -0.441176470588240;
            invA0(3, 3) = +3.999999999999998;
            invA0(3, 4) = +0.000000000000000;
            invA0(4, 0) = +12.333333333333645;
            invA0(4, 1) = +17.166666666666643;
            invA0(4, 2) = -137.500000000000171;
            invA0(4, 3) = +113.333333333333329;
            invA0(4, 4) = +4.000000000000000;
            /* --- b --- */
            b0(0) = +1.041666666666667;
            b0(1) = -1.020833333333333;
            b0(2) = +7.812500000000000;
            b0(3) = -7.083333333333333;
            b0(4) = +0.250000000000000;
            /* --- c --- */
            c0(0) = +0.250000000000000;
            c0(1) = +0.750000000000000;
            c0(2) = +0.550000000000000;
            c0(3) = +0.500000000000000;
            c0(4) = +1.000000000000000;
            /* --- d --- */
            d0(0) = +0.000000000000000;
            d0(1) = +0.000000000000000;
            d0(2) = +0.000000000000000;
            d0(3) = +0.000000000000000;
            d0(4) = +1.000000000000000;
            /* --- Q --- */
            Q0(0, 0) = +0.000000000000000;
            Q0(0, 1) = +0.000000000000000;
            Q0(0, 2) = +0.000000000000000;
            Q0(0, 3) = +0.000000000000000;
            Q0(0, 4) = -1.000000000000000;
            Q0(1, 0) = +0.000000000000000;
            Q0(1, 1) = +0.000000000000000;
            Q0(1, 2) = +0.000000000000000;
            Q0(1, 3) = +1.000000000000000;
            Q0(1, 4) = +0.000000000000000;
            Q0(2, 0) = +0.000000000000000;
            Q0(2, 1) = +0.000000000000000;
            Q0(2, 2) = -1.000000000000000;
            Q0(2, 3) = +0.000000000000000;
            Q0(2, 4) = +0.000000000000000;
            Q0(3, 0) = +0.000000000000000;
            Q0(3, 1) = +1.000000000000000;
            Q0(3, 2) = +0.000000000000000;
            Q0(3, 3) = +0.000000000000000;
            Q0(3, 4) = +0.000000000000000;
            Q0(4, 0) = -1.000000000000000;
            Q0(4, 1) = +0.000000000000000;
            Q0(4, 2) = +0.000000000000000;
            Q0(4, 3) = +0.000000000000000;
            Q0(4, 4) = +0.000000000000000;
            /* --- R --- */
            R0(0, 0) = +4.000000000000000;
            R0(0, 1) = -113.333333333333329;
            R0(0, 2) = -137.500000000000000;
            R0(0, 3) = -17.166666666666671;
            R0(0, 4) = +12.333333333333357;
            R0(1, 0) = +0.000000000000000;
            R0(1, 1) = +4.000000000000000;
            R0(1, 2) = +0.441176470588235;
            R0(1, 3) = +0.735294117647059;
            R0(1, 4) = +5.235294117647059;
            R0(2, 0) = +0.000000000000000;
            R0(2, 1) = +0.000000000000000;
            R0(2, 2) = +4.000000000000000;
            R0(2, 3) = -0.640000000000000;
            R0(2, 4) = -6.720000000000001;
            R0(3, 0) = +0.000000000000000;
            R0(3, 1) = +0.000000000000000;
            R0(3, 2) = +0.000000000000000;
            R0(3, 3) = +4.000000000000000;
            R0(3, 4) = +8.000000000000000;
            R0(4, 0) = +0.000000000000000;
            R0(4, 1) = +0.000000000000000;
            R0(4, 2) = +0.000000000000000;
            R0(4, 3) = +0.000000000000000;
            R0(4, 4) = +4.000000000000000;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 1;
            R0_block_sizes[1] = 1;
            R0_block_sizes[2] = 1;
            R0_block_sizes[3] = 1;
            R0_block_sizes[4] = 1;
            break;

        // 1-stage 2nd-order Gauss--Legendre
        case Type::Gauss2:
            is_imex = false;
            s = 1;
            s_eff = 1;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.500000000000000;
            /* --- inv(A) --- */
            invA0(0, 0) = +2.000000000000000;
            /* --- b --- */
            b0(0) = +1.000000000000000;
            /* --- c --- */
            c0(0) = +0.500000000000000;
            /* --- d --- */
            d0(0) = +2.000000000000000;
            /* --- Q --- */
            Q0(0, 0) = +1.000000000000000;
            /* --- R --- */
            R0(0, 0) = +2.000000000000000;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 1;
            break;

        // 2-stage 4th-order Gauss--Legendre
        case Type::Gauss4:
            is_imex = false;
            s = 2;
            s_eff = 1;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.250000000000000;
            A0(0, 1) = -0.038675134594813;
            A0(1, 0) = +0.538675134594813;
            A0(1, 1) = +0.250000000000000;
            /* --- inv(A) --- */
            invA0(0, 0) = +3.000000000000000;
            invA0(0, 1) = +0.464101615137754;
            invA0(1, 0) = -6.464101615137755;
            invA0(1, 1) = +3.000000000000000;
            /* --- b --- */
            b0(0) = +0.500000000000000;
            b0(1) = +0.500000000000000;
            /* --- c --- */
            c0(0) = +0.211324865405187;
            c0(1) = +0.788675134594813;
            /* --- d --- */
            d0(0) = -1.732050807568877;
            d0(1) = +1.732050807568877;
            /* --- Q --- */
            Q0(0, 0) = +1.000000000000000;
            Q0(0, 1) = +0.000000000000000;
            Q0(1, 0) = +0.000000000000000;
            Q0(1, 1) = +1.000000000000000;
            /* --- R --- */
            R0(0, 0) = +3.000000000000000;
            R0(0, 1) = +0.464101615137754;
            R0(1, 0) = -6.464101615137755;
            R0(1, 1) = +3.000000000000000;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 2;
            break;

        // 3-stage 6th-order Gauss--Legendre
        case Type::Gauss6:
            is_imex = false;
            s = 3;
            s_eff = 2;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.138888888888889;
            A0(0, 1) = -0.035976667524939;
            A0(0, 2) = +0.009789444015308;
            A0(1, 0) = +0.300263194980865;
            A0(1, 1) = +0.222222222222222;
            A0(1, 2) = -0.022485417203087;
            A0(2, 0) = +0.267988333762469;
            A0(2, 1) = +0.480421111969383;
            A0(2, 2) = +0.138888888888889;
            /* --- inv(A) --- */
            invA0(0, 0) = +4.999999999999997;
            invA0(0, 1) = +1.163977794943223;
            invA0(0, 2) = -0.163977794943222;
            invA0(1, 0) = -5.727486121839513;
            invA0(1, 1) = +2.000000000000000;
            invA0(1, 2) = +0.727486121839514;
            invA0(2, 0) = +10.163977794943225;
            invA0(2, 1) = -9.163977794943223;
            invA0(2, 2) = +5.000000000000001;
            /* --- b --- */
            b0(0) = +0.277777777777778;
            b0(1) = +0.444444444444444;
            b0(2) = +0.277777777777778;
            /* --- c --- */
            c0(0) = +0.112701665379258;
            c0(1) = +0.500000000000000;
            c0(2) = +0.887298334620742;
            /* --- d --- */
            d0(0) = +1.666666666666668;
            d0(1) = -1.333333333333334;
            d0(2) = +1.666666666666667;
            /* --- Q --- */
            Q0(0, 0) = -0.083238672942822;
            Q0(0, 1) = -0.181066379097451;
            Q0(0, 2) = +0.979941982816971;
            Q0(1, 0) = +0.060321530401366;
            Q0(1, 1) = +0.980635889214054;
            Q0(1, 2) = +0.186318452535969;
            Q0(2, 0) = +0.994702285257631;
            Q0(2, 1) = -0.074620500841923;
            Q0(2, 2) = +0.070704628966890;
            /* --- R --- */
            R0(0, 0) = +3.677814645373911;
            R0(0, 1) = -10.983730977131060;
            R0(0, 2) = +7.822707869262245;
            R0(1, 0) = +1.120876889086220;
            R0(1, 1) = +3.677814645373911;
            R0(1, 2) = -6.654600663998786;
            R0(2, 0) = +0.000000000000000;
            R0(2, 1) = +0.000000000000000;
            R0(2, 2) = +4.644370709252176;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 2;
            R0_block_sizes[1] = 1;
            break;

        // 4-stage 8th-order Gauss--Legendre
        case Type::Gauss8:
            is_imex = false;
            s = 4;
            s_eff = 2;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.086963711284363;
            A0(0, 1) = -0.026604180084999;
            A0(0, 2) = +0.012627462689405;
            A0(0, 3) = -0.003555149685796;
            A0(1, 0) = +0.188118117499868;
            A0(1, 1) = +0.163036288715637;
            A0(1, 2) = -0.027880428602471;
            A0(1, 3) = +0.006735500594538;
            A0(2, 0) = +0.167191921974189;
            A0(2, 1) = +0.353953006033744;
            A0(2, 2) = +0.163036288715637;
            A0(2, 3) = -0.014190694931141;
            A0(3, 0) = +0.177482572254523;
            A0(3, 1) = +0.313445114741868;
            A0(3, 2) = +0.352676757516272;
            A0(3, 3) = +0.086963711284363;
            /* --- inv(A) --- */
            invA0(0, 0) = +7.738612787525829;
            invA0(0, 1) = +2.045089650303909;
            invA0(0, 2) = -0.437070802395799;
            invA0(0, 3) = +0.086644023503262;
            invA0(1, 0) = -7.201340999706891;
            invA0(1, 1) = +2.261387212474169;
            invA0(1, 2) = +1.448782034533682;
            invA0(1, 3) = -0.233133981212417;
            invA0(2, 0) = +6.343622218624971;
            invA0(2, 1) = -5.971556459482020;
            invA0(2, 2) = +2.261387212474169;
            invA0(2, 3) = +1.090852762294336;
            invA0(3, 0) = -15.563869598554923;
            invA0(3, 1) = +11.892783878056845;
            invA0(3, 2) = -13.500802725964952;
            invA0(3, 3) = +7.738612787525829;
            /* --- b --- */
            b0(0) = +0.173927422568727;
            b0(1) = +0.326072577431273;
            b0(2) = +0.326072577431273;
            b0(3) = +0.173927422568727;
            /* --- c --- */
            c0(0) = +0.069431844202974;
            c0(1) = +0.330009478207572;
            c0(2) = +0.669990521792428;
            c0(3) = +0.930568155797026;
            /* --- d --- */
            d0(0) = -1.640705321739257;
            d0(1) = +1.214393969798579;
            d0(2) = -1.214393969798578;
            d0(3) = +1.640705321739257;
            /* --- Q --- */
            Q0(0, 0) = +0.048240345831208;
            Q0(0, 1) = +0.022512328854129;
            Q0(0, 2) = -0.336698980990652;
            Q0(0, 3) = -0.940106302650665;
            Q0(1, 0) = -0.122245946830287;
            Q0(1, 1) = +0.092517479973163;
            Q0(1, 2) = +0.929016942971458;
            Q0(1, 3) = -0.336784744391653;
            Q0(2, 0) = +0.111699705302275;
            Q0(2, 1) = -0.987901978189380;
            Q0(2, 2) = +0.094333738332995;
            Q0(2, 3) = -0.051710764227736;
            Q0(3, 0) = +0.985013691962216;
            Q0(3, 1) = +0.122406668276236;
            Q0(3, 2) = +0.121088652168346;
            Q0(3, 3) = +0.010108042566567;
            /* --- R --- */
            R0(0, 0) = +4.207578794359268;
            R0(0, 1) = +14.740509080125308;
            R0(0, 2) = +14.291665730718535;
            R0(0, 3) = +9.670422106794417;
            R0(1, 0) = -1.916316624018728;
            R0(1, 1) = +4.207578794359268;
            R0(1, 2) = +9.614204997121385;
            R0(1, 3) = +5.775112524448543;
            R0(2, 0) = +0.000000000000000;
            R0(2, 1) = +0.000000000000000;
            R0(2, 2) = +5.792421205640723;
            R0(2, 3) = +9.181556269430962;
            R0(3, 0) = +0.000000000000000;
            R0(3, 1) = +0.000000000000000;
            R0(3, 2) = -0.327654707902998;
            R0(3, 3) = +5.792421205640723;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 2;
            R0_block_sizes[1] = 2;
            break;

        // 5-stage 10th-order Gauss--Legendre
        case Type::Gauss10:
            is_imex = false;
            s = 5;
            s_eff = 3;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.059231721264047;
            A0(0, 1) = -0.019570364359076;
            A0(0, 2) = +0.011254400818643;
            A0(0, 3) = -0.005593793660812;
            A0(0, 4) = +0.001588112967866;
            A0(1, 0) = +0.128151005670045;
            A0(1, 1) = +0.119657167624842;
            A0(1, 2) = -0.024592114619642;
            A0(1, 3) = +0.010318280670683;
            A0(1, 4) = -0.002768994398770;
            A0(2, 0) = +0.113776288004225;
            A0(2, 1) = +0.260004651680642;
            A0(2, 2) = +0.142222222222222;
            A0(2, 3) = -0.020690316430958;
            A0(2, 4) = +0.004687154523870;
            A0(3, 0) = +0.121232436926864;
            A0(3, 1) = +0.228996054579000;
            A0(3, 2) = +0.309036559064087;
            A0(3, 3) = +0.119657167624842;
            A0(3, 4) = -0.009687563141951;
            A0(4, 0) = +0.116875329560229;
            A0(4, 1) = +0.244908128910495;
            A0(4, 2) = +0.273190043625802;
            A0(4, 3) = +0.258884699608759;
            A0(4, 4) = +0.059231721264047;
            /* --- inv(A) --- */
            invA0(0, 0) = +11.183300132670379;
            invA0(0, 1) = +3.131312162011809;
            invA0(0, 2) = -0.758731795980806;
            invA0(0, 3) = +0.239101223353686;
            invA0(0, 4) = -0.054314760565339;
            invA0(1, 0) = -9.447599601516151;
            invA0(1, 1) = +2.816699867329624;
            invA0(1, 2) = +2.217886332274817;
            invA0(1, 3) = -0.557122620293796;
            invA0(1, 4) = +0.118357949604667;
            invA0(2, 0) = +6.420116503559337;
            invA0(2, 1) = -6.220120454669752;
            invA0(2, 2) = +2.000000000000000;
            invA0(2, 3) = +1.865995288831779;
            invA0(2, 4) = -0.315991337721364;
            invA0(3, 0) = -8.015920784810973;
            invA0(3, 1) = +6.190522354953043;
            invA0(3, 2) = -7.393116276382382;
            invA0(3, 3) = +2.816699867329624;
            invA0(3, 4) = +1.550036766309846;
            invA0(4, 0) = +22.420915025906101;
            invA0(4, 1) = -16.193390239999243;
            invA0(4, 2) = +15.415443221569852;
            invA0(4, 3) = -19.085601178657356;
            invA0(4, 4) = +11.183300132670379;
            /* --- b --- */
            b0(0) = +0.118463442528095;
            b0(1) = +0.239314335249683;
            b0(2) = +0.284444444444444;
            b0(3) = +0.239314335249683;
            b0(4) = +0.118463442528095;
            /* --- c --- */
            c0(0) = +0.046910077030668;
            c0(1) = +0.230765344947158;
            c0(2) = +0.500000000000000;
            c0(3) = +0.769234655052841;
            c0(4) = +0.953089922969332;
            /* --- d --- */
            d0(0) = +1.627766710890126;
            d0(1) = -1.161100044223459;
            d0(2) = +1.066666666666666;
            d0(3) = -1.161100044223459;
            d0(4) = +1.627766710890126;
            /* --- Q --- */
            Q0(0, 0) = +0.052286373550732;
            Q0(0, 1) = +0.017761459146135;
            Q0(0, 2) = +0.107901918652109;
            Q0(0, 3) = -0.882638337196087;
            Q0(0, 4) = +0.454155708290488;
            Q0(1, 0) = -0.087217337438407;
            Q0(1, 1) = -0.051247596714227;
            Q0(1, 2) = -0.222975377135070;
            Q0(1, 3) = -0.469846600315604;
            Q0(1, 4) = -0.848111415584363;
            Q0(2, 0) = -0.005226019290336;
            Q0(2, 1) = +0.155957034844084;
            Q0(2, 2) = +0.954541442335525;
            Q0(2, 3) = -0.011034361806211;
            Q0(2, 4) = -0.253730111986263;
            Q0(3, 0) = +0.981199881220355;
            Q0(3, 1) = -0.167084318758466;
            Q0(3, 2) = +0.007191508152453;
            Q0(3, 3) = +0.006182333497419;
            Q0(3, 4) = -0.096123277520073;
            Q0(4, 0) = -0.163947409270801;
            Q0(4, 1) = -0.972017720058475;
            Q0(4, 2) = +0.165644421026408;
            Q0(4, 3) = +0.005810292217529;
            Q0(4, 4) = +0.028826466534928;
            /* --- R --- */
            R0(0, 0) = +4.649348606363301;
            R0(0, 1) = -2.866988395237885;
            R0(0, 2) = -12.612472186300799;
            R0(0, 3) = +5.040226491530089;
            R0(0, 4) = -10.015206098251161;
            R0(1, 0) = +17.791777209507831;
            R0(1, 1) = +4.649348606363301;
            R0(1, 2) = -19.736222014661173;
            R0(1, 3) = +10.353647543145909;
            R0(1, 4) = -18.748802010104690;
            R0(2, 0) = +0.000000000000000;
            R0(2, 1) = +0.000000000000000;
            R0(2, 2) = +7.293477190659350;
            R0(2, 3) = -7.436145789040616;
            R0(2, 4) = +12.655118452703233;
            R0(3, 0) = +0.000000000000000;
            R0(3, 1) = +0.000000000000000;
            R0(3, 2) = +0.000000000000000;
            R0(3, 3) = +6.703912798307031;
            R0(3, 4) = +1.090191404805731;
            R0(4, 0) = +0.000000000000000;
            R0(4, 1) = +0.000000000000000;
            R0(4, 2) = +0.000000000000000;
            R0(4, 3) = -11.142516068523712;
            R0(4, 4) = +6.703912798307031;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 2;
            R0_block_sizes[1] = 1;
            R0_block_sizes[2] = 2;
            break;

        // 2-stage 3rd-order Radau IIA
        case Type::RadauIIA3:
            is_imex = false;
            s = 2;
            s_eff = 1;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.416666666666667;
            A0(0, 1) = -0.083333333333333;
            A0(1, 0) = +0.750000000000000;
            A0(1, 1) = +0.250000000000000;
            /* --- inv(A) --- */
            invA0(0, 0) = +1.500000000000000;
            invA0(0, 1) = +0.500000000000000;
            invA0(1, 0) = -4.500000000000000;
            invA0(1, 1) = +2.500000000000000;
            /* --- b --- */
            b0(0) = +0.750000000000000;
            b0(1) = +0.250000000000000;
            /* --- c --- */
            c0(0) = +0.333333333333333;
            c0(1) = +1.000000000000000;
            /* --- d --- */
            d0(0) = -0.000000000000000;
            d0(1) = +1.000000000000000;
            /* --- Q --- */
            Q0(0, 0) = +0.992507556682903;
            Q0(0, 1) = +0.122183263695704;
            Q0(1, 0) = -0.122183263695704;
            Q0(1, 1) = +0.992507556682903;
            /* --- R --- */
            R0(0, 0) = +2.000000000000000;
            R0(0, 1) = +0.438447187191170;
            R0(1, 0) = -4.561552812808831;
            R0(1, 1) = +2.000000000000000;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 2;
            break;

        // 3-stage 5th-order Radau IIA
        case Type::RadauIIA5:
            is_imex = false;
            s = 3;
            s_eff = 2;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.196815477223660;
            A0(0, 1) = -0.065535425850198;
            A0(0, 2) = +0.023770974348220;
            A0(1, 0) = +0.394424314739087;
            A0(1, 1) = +0.292073411665228;
            A0(1, 2) = -0.041548752125998;
            A0(2, 0) = +0.376403062700467;
            A0(2, 1) = +0.512485826188422;
            A0(2, 2) = +0.111111111111111;
            /* --- inv(A) --- */
            invA0(0, 0) = +3.224744871391589;
            invA0(0, 1) = +1.167840084690405;
            invA0(0, 2) = -0.253197264742181;
            invA0(1, 0) = -3.567840084690406;
            invA0(1, 1) = +0.775255128608411;
            invA0(1, 2) = +1.053197264742181;
            invA0(2, 0) = +5.531972647421809;
            invA0(2, 1) = -7.531972647421810;
            invA0(2, 2) = +5.000000000000001;
            /* --- b --- */
            b0(0) = +0.376403062700467;
            b0(1) = +0.512485826188422;
            b0(2) = +0.111111111111111;
            /* --- c --- */
            c0(0) = +0.155051025721682;
            c0(1) = +0.644948974278318;
            c0(2) = +1.000000000000000;
            /* --- d --- */
            d0(0) = +0.000000000000000;
            d0(1) = +0.000000000000000;
            d0(2) = +1.000000000000000;
            /* --- Q --- */
            Q0(0, 0) = +0.138665108751908;
            Q0(0, 1) = +0.046278149309489;
            Q0(0, 2) = +0.989257459163847;
            Q0(1, 0) = -0.229641242351741;
            Q0(1, 1) = -0.970178886551833;
            Q0(1, 2) = +0.077574660168093;
            Q0(2, 0) = -0.963346711950568;
            Q0(2, 1) = +0.237931210616714;
            Q0(2, 2) = +0.123902589111344;
            /* --- R --- */
            R0(0, 0) = +2.681082873627751;
            R0(0, 1) = -8.423875798085390;
            R0(0, 2) = -4.088577813059627;
            R0(1, 0) = +1.104613199852199;
            R0(1, 1) = +2.681082873627751;
            R0(1, 2) = +4.700152006343942;
            R0(2, 0) = +0.000000000000000;
            R0(2, 1) = +0.000000000000000;
            R0(2, 2) = +3.637834252744495;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 2;
            R0_block_sizes[1] = 1;
            break;

        // 4-stage 7th-order Radau IIA
        case Type::RadauIIA7:
            is_imex = false;
            s = 4;
            s_eff = 2;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.112999479323156;
            A0(0, 1) = -0.040309220723522;
            A0(0, 2) = +0.025802377420336;
            A0(0, 3) = -0.009904676507266;
            A0(1, 0) = +0.234383995747400;
            A0(1, 1) = +0.206892573935359;
            A0(1, 2) = -0.047857128048541;
            A0(1, 3) = +0.016047422806516;
            A0(2, 0) = +0.216681784623250;
            A0(2, 1) = +0.406123263867373;
            A0(2, 2) = +0.189036518170056;
            A0(2, 3) = -0.024182104899833;
            A0(3, 0) = +0.220462211176768;
            A0(3, 1) = +0.388193468843172;
            A0(3, 2) = +0.328844319980060;
            A0(3, 3) = +0.062500000000000;
            /* --- inv(A) --- */
            invA0(0, 0) = +5.644107875950090;
            invA0(0, 1) = +1.923507277054713;
            invA0(0, 2) = -0.585901482103817;
            invA0(0, 3) = +0.173878352574246;
            invA0(1, 0) = -5.049214638391410;
            invA0(1, 1) = +1.221100028894693;
            invA0(1, 2) = +1.754680988760837;
            invA0(1, 3) = -0.434791461212582;
            invA0(2, 0) = +3.492466158625437;
            invA0(2, 1) = -3.984517895782496;
            invA0(2, 2) = +0.634792095155218;
            invA0(2, 3) = +1.822137598434254;
            invA0(3, 0) = -6.923488256445451;
            invA0(3, 1) = +6.595237669628144;
            invA0(3, 2) = -12.171749413182692;
            invA0(3, 3) = +8.500000000000002;
            /* --- b --- */
            b0(0) = +0.220462211176768;
            b0(1) = +0.388193468843172;
            b0(2) = +0.328844319980060;
            b0(3) = +0.062500000000000;
            /* --- c --- */
            c0(0) = +0.088587959512704;
            c0(1) = +0.409466864440735;
            c0(2) = +0.787659461760847;
            c0(3) = +1.000000000000000;
            /* --- d --- */
            d0(0) = -0.000000000000000;
            d0(1) = +0.000000000000000;
            d0(2) = -0.000000000000000;
            d0(3) = +1.000000000000000;
            /* --- Q --- */
            Q0(0, 0) = +0.054570292994775;
            Q0(0, 1) = +0.125683294495023;
            Q0(0, 2) = +0.956633137728588;
            Q0(0, 3) = -0.257058033149907;
            Q0(1, 0) = -0.177667119738735;
            Q0(1, 1) = -0.126991262238752;
            Q0(1, 2) = +0.278165559200843;
            Q0(1, 3) = +0.935377750191457;
            Q0(2, 0) = +0.321254131229142;
            Q0(2, 1) = -0.939193753944121;
            Q0(2, 2) = +0.080747339995071;
            Q0(2, 3) = -0.090502722634621;
            Q0(3, 0) = +0.928575393198859;
            Q0(3, 1) = +0.293243962175244;
            Q0(3, 2) = -0.030932645502120;
            Q0(3, 3) = +0.225386089268147;
            /* --- R --- */
            R0(0, 0) = +3.212806896871531;
            R0(0, 1) = +12.141571045775553;
            R0(0, 2) = -3.796586217388935;
            R0(0, 3) = +8.446761444650136;
            R0(1, 0) = -1.876393389274784;
            R0(1, 1) = +3.212806896871531;
            R0(1, 2) = -2.571864526766571;
            R0(1, 3) = +7.005683821998450;
            R0(2, 0) = +0.000000000000000;
            R0(2, 1) = +0.000000000000000;
            R0(2, 2) = +4.787193103128468;
            R0(2, 3) = +0.344651300061515;
            R0(3, 0) = +0.000000000000000;
            R0(3, 1) = +0.000000000000000;
            R0(3, 2) = -7.128893223626639;
            R0(3, 3) = +4.787193103128468;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 2;
            R0_block_sizes[1] = 2;
            break;

        // 5-stage 9th-order Radau IIA
        case Type::RadauIIA9:
            is_imex = false;
            s = 5;
            s_eff = 3;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.072998864317903;
            A0(0, 1) = -0.026735331107946;
            A0(0, 2) = +0.018676929763984;
            A0(0, 3) = -0.012879106093306;
            A0(0, 4) = +0.005042839233882;
            A0(1, 0) = +0.153775231479182;
            A0(1, 1) = +0.146214867847494;
            A0(1, 2) = -0.036444568905128;
            A0(1, 3) = +0.021233063119305;
            A0(1, 4) = -0.007935579902729;
            A0(2, 0) = +0.140063045684810;
            A0(2, 1) = +0.298967129491283;
            A0(2, 2) = +0.167585070135249;
            A0(2, 3) = -0.033969101686618;
            A0(2, 4) = +0.010944288744192;
            A0(3, 0) = +0.144894308109535;
            A0(3, 1) = +0.276500068760159;
            A0(3, 2) = +0.325797922910421;
            A0(3, 3) = +0.128756753254910;
            A0(3, 4) = -0.015708917378805;
            A0(4, 0) = +0.143713560791226;
            A0(4, 1) = +0.281356015149462;
            A0(4, 2) = +0.311826522975741;
            A0(4, 3) = +0.223103901083571;
            A0(4, 4) = +0.040000000000000;
            /* --- inv(A) --- */
            invA0(0, 0) = +8.755923977938361;
            invA0(0, 1) = +2.891942615380119;
            invA0(0, 2) = -0.875186396200266;
            invA0(0, 3) = +0.399705207939967;
            invA0(0, 4) = -0.133706163849216;
            invA0(1, 0) = -7.161380720145386;
            invA0(1, 1) = +1.806077724083645;
            invA0(1, 2) = +2.363797176068608;
            invA0(1, 3) = -0.865900780283135;
            invA0(1, 4) = +0.274338077775194;
            invA0(2, 0) = +4.122165246243370;
            invA0(2, 1) = -4.496017125813395;
            invA0(2, 2) = +0.856765245397177;
            invA0(2, 3) = +2.518320949211065;
            invA0(2, 4) = -0.657062757134360;
            invA0(3, 0) = -3.878663219724007;
            invA0(3, 1) = +3.393151918064955;
            invA0(3, 2) = -5.188340906407188;
            invA0(3, 3) = +0.581233052580817;
            invA0(3, 4) = +2.809983655279712;
            invA0(4, 0) = +8.412424223594297;
            invA0(4, 1) = -6.970256116656662;
            invA0(4, 2) = +8.777114204150472;
            invA0(4, 3) = -18.219282311088101;
            invA0(4, 4) = +13.000000000000000;
            /* --- b --- */
            b0(0) = +0.143713560791226;
            b0(1) = +0.281356015149462;
            b0(2) = +0.311826522975741;
            b0(3) = +0.223103901083571;
            b0(4) = +0.040000000000000;
            /* --- c --- */
            c0(0) = +0.057104196114518;
            c0(1) = +0.276843013638124;
            c0(2) = +0.583590432368917;
            c0(3) = +0.860240135656219;
            c0(4) = +1.000000000000000;
            /* --- d --- */
            d0(0) = +0.000000000000000;
            d0(1) = -0.000000000000000;
            d0(2) = +0.000000000000000;
            d0(3) = +0.000000000000000;
            d0(4) = +1.000000000000000;
            /* --- Q --- */
            Q0(0, 0) = +0.104376011865108;
            Q0(0, 1) = +0.002753215699337;
            Q0(0, 2) = +0.034631183267103;
            Q0(0, 3) = +0.423574187128014;
            Q0(0, 4) = +0.899157192650324;
            Q0(1, 0) = -0.216061922297273;
            Q0(1, 1) = -0.032215375782065;
            Q0(1, 2) = -0.064584428263704;
            Q0(1, 3) = -0.869905614769537;
            Q0(1, 4) = +0.437461413515194;
            Q0(2, 0) = +0.291316048441817;
            Q0(2, 1) = +0.192522319893028;
            Q0(2, 2) = +0.925248363516386;
            Q0(2, 3) = -0.148275233623860;
            Q0(2, 4) = -0.000192788196279;
            Q0(3, 0) = +0.869441565532768;
            Q0(3, 1) = -0.401017866194354;
            Q0(3, 2) = -0.220183152562227;
            Q0(3, 3) = -0.186450946621164;
            Q0(3, 4) = -0.003385106894551;
            Q0(4, 0) = -0.318793378106442;
            Q0(4, 1) = -0.895027606670579;
            Q0(4, 2) = +0.300107277334918;
            Q0(4, 3) = +0.084259296158572;
            Q0(4, 4) = -0.011504715315962;
            /* --- R --- */
            R0(0, 0) = +3.655694325463577;
            R0(0, 1) = -2.761497758898925;
            R0(0, 2) = -9.607413444485667;
            R0(0, 3) = -4.682534851529246;
            R0(0, 4) = -0.500335166972660;
            R0(1, 0) = +15.506256512451463;
            R0(1, 1) = +3.655694325463577;
            R0(1, 2) = -13.213023448159136;
            R0(1, 3) = -8.868956774490409;
            R0(1, 4) = -2.602235394882079;
            R0(2, 0) = +0.000000000000000;
            R0(2, 1) = +0.000000000000000;
            R0(2, 2) = +6.286704751729317;
            R0(2, 3) = +9.642681278729382;
            R0(2, 4) = +4.066371799790950;
            R0(3, 0) = +0.000000000000000;
            R0(3, 1) = +0.000000000000000;
            R0(3, 2) = +0.000000000000000;
            R0(3, 3) = +5.700953298671759;
            R0(3, 4) = +9.280382169459997;
            R0(4, 0) = +0.000000000000000;
            R0(4, 1) = +0.000000000000000;
            R0(4, 2) = +0.000000000000000;
            R0(4, 3) = -1.110493623682757;
            R0(4, 4) = +5.700953298671759;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 2;
            R0_block_sizes[1] = 1;
            R0_block_sizes[2] = 2;
            break;

        // 2-stage 2nd-order Lobatto IIIC
        case Type::LobIIIC2:
            is_imex = false;
            s = 2;
            s_eff = 1;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.500000000000000;
            A0(0, 1) = -0.500000000000000;
            A0(1, 0) = +0.500000000000000;
            A0(1, 1) = +0.500000000000000;
            /* --- inv(A) --- */
            invA0(0, 0) = +1.000000000000000;
            invA0(0, 1) = +1.000000000000000;
            invA0(1, 0) = -1.000000000000000;
            invA0(1, 1) = +1.000000000000000;
            /* --- b --- */
            b0(0) = +0.500000000000000;
            b0(1) = +0.500000000000000;
            /* --- c --- */
            c0(0) = +0.000000000000000;
            c0(1) = +1.000000000000000;
            /* --- d --- */
            d0(0) = +0.000000000000000;
            d0(1) = +1.000000000000000;
            /* --- Q --- */
            Q0(0, 0) = +1.000000000000000;
            Q0(0, 1) = +0.000000000000000;
            Q0(1, 0) = +0.000000000000000;
            Q0(1, 1) = +1.000000000000000;
            /* --- R --- */
            R0(0, 0) = +1.000000000000000;
            R0(0, 1) = +1.000000000000000;
            R0(1, 0) = -1.000000000000000;
            R0(1, 1) = +1.000000000000000;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 2;
            break;

        // 3-stage 4th-order Lobatto IIIC
        case Type::LobIIIC4:
            is_imex = false;
            s = 3;
            s_eff = 2;
            SizeData();
            
            /* --- A --- */
            A0(0, 0) = +0.166666666666667;
            A0(0, 1) = -0.333333333333333;
            A0(0, 2) = +0.166666666666667;
            A0(1, 0) = +0.166666666666667;
            A0(1, 1) = +0.416666666666667;
            A0(1, 2) = -0.083333333333333;
            A0(2, 0) = +0.166666666666667;
            A0(2, 1) = +0.666666666666667;
            A0(2, 2) = +0.166666666666667;
            /* --- inv(A) --- */
            invA0(0, 0) = +3.000000000000000;
            invA0(0, 1) = +4.000000000000000;
            invA0(0, 2) = -1.000000000000000;
            invA0(1, 0) = -1.000000000000000;
            invA0(1, 1) = +0.000000000000000;
            invA0(1, 2) = +1.000000000000000;
            invA0(2, 0) = +1.000000000000000;
            invA0(2, 1) = -4.000000000000000;
            invA0(2, 2) = +3.000000000000000;
            /* --- b --- */
            b0(0) = +0.166666666666667;
            b0(1) = +0.666666666666667;
            b0(2) = +0.166666666666667;
            /* --- c --- */
            c0(0) = +0.000000000000000;
            c0(1) = +0.500000000000000;
            c0(2) = +1.000000000000000;
            /* --- d --- */
            d0(0) = +0.000000000000000;
            d0(1) = -0.000000000000000;
            d0(2) = +1.000000000000000;
            /* --- Q --- */
            Q0(0, 0) = -0.575784070910962;
            Q0(0, 1) = -0.375422043049930;
            Q0(0, 2) = +0.726313288655396;
            Q0(1, 0) = +0.270505426506658;
            Q0(1, 1) = +0.750844086099862;
            Q0(1, 2) = +0.602544581420590;
            Q0(2, 0) = +0.771556555228230;
            Q0(2, 1) = -0.543407257920869;
            Q0(2, 2) = +0.330770364638777;
            /* --- R --- */
            R0(0, 0) = +1.687091590520766;
            R0(0, 1) = -5.303878657262186;
            R0(0, 2) = -3.092458332654278;
            R0(1, 0) = +1.186628772049857;
            R0(1, 1) = +1.687091590520766;
            R0(1, 2) = -1.519873274599652;
            R0(2, 0) = +0.000000000000000;
            R0(2, 1) = +0.000000000000000;
            R0(2, 2) = +2.625816818958467;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 2;
            R0_block_sizes[1] = 1;
            break;

        // 4-stage 6th-order Lobatto IIIC
        case Type::LobIIIC6:
            is_imex = false;
            s = 4;
            s_eff = 2;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.083333333333333;
            A0(0, 1) = -0.186338998124982;
            A0(0, 2) = +0.186338998124982;
            A0(0, 3) = -0.083333333333333;
            A0(1, 0) = +0.083333333333333;
            A0(1, 1) = +0.250000000000000;
            A0(1, 2) = -0.094207930708309;
            A0(1, 3) = +0.037267799624996;
            A0(2, 0) = +0.083333333333333;
            A0(2, 1) = +0.427541264041642;
            A0(2, 2) = +0.250000000000000;
            A0(2, 3) = -0.037267799624996;
            A0(3, 0) = +0.083333333333333;
            A0(3, 1) = +0.416666666666667;
            A0(3, 2) = +0.416666666666667;
            A0(3, 3) = +0.083333333333333;
            /* --- inv(A) --- */
            invA0(0, 0) = +5.999999999999999;
            invA0(0, 1) = +8.090169943749475;
            invA0(0, 2) = -3.090169943749473;
            invA0(0, 3) = +0.999999999999999;
            invA0(1, 0) = -1.618033988749895;
            invA0(1, 1) = -0.000000000000000;
            invA0(1, 2) = +2.236067977499789;
            invA0(1, 3) = -0.618033988749895;
            invA0(2, 0) = +0.618033988749895;
            invA0(2, 1) = -2.236067977499789;
            invA0(2, 2) = -0.000000000000001;
            invA0(2, 3) = +1.618033988749895;
            invA0(3, 0) = -1.000000000000000;
            invA0(3, 1) = +3.090169943749475;
            invA0(3, 2) = -8.090169943749475;
            invA0(3, 3) = +6.000000000000000;
            /* --- b --- */
            b0(0) = +0.083333333333333;
            b0(1) = +0.416666666666667;
            b0(2) = +0.416666666666667;
            b0(3) = +0.083333333333333;
            /* --- c --- */
            c0(0) = +0.000000000000000;
            c0(1) = +0.276393202250021;
            c0(2) = +0.723606797749979;
            c0(3) = +1.000000000000000;
            /* --- d --- */
            d0(0) = -0.000000000000000;
            d0(1) = +0.000000000000000;
            d0(2) = +0.000000000000000;
            d0(3) = +1.000000000000000;
            /* --- Q --- */
            Q0(0, 0) = +0.106073254812433;
            Q0(0, 1) = -0.856299374198130;
            Q0(0, 2) = -0.458004872738390;
            Q0(0, 3) = +0.213848972195978;
            Q0(1, 0) = -0.065022767538930;
            Q0(1, 1) = -0.038030973632481;
            Q0(1, 2) = +0.467333521122043;
            Q0(1, 3) = +0.880866087882725;
            Q0(2, 0) = -0.300594214556409;
            Q0(2, 1) = -0.510458447047536;
            Q0(2, 2) = +0.692545715189596;
            Q0(2, 3) = -0.411650002290251;
            Q0(3, 0) = -0.945602253852162;
            Q0(3, 1) = +0.068827324735684;
            Q0(3, 2) = -0.303663216336939;
            Q0(3, 3) = +0.094275277370812;
            /* --- R --- */
            R0(0, 0) = +3.779019967010190;
            R0(0, 1) = -5.310199803237381;
            R0(0, 2) = +5.458377049964716;
            R0(0, 3) = -4.426194015356730;
            R0(1, 0) = +0.358722328488004;
            R0(1, 1) = +3.779019967010190;
            R0(1, 2) = +1.662610499364627;
            R0(1, 3) = -7.019736960225162;
            R0(2, 0) = +0.000000000000000;
            R0(2, 1) = +0.000000000000000;
            R0(2, 2) = +2.220980032989806;
            R0(2, 3) = -8.208087489090918;
            R0(3, 0) = +0.000000000000000;
            R0(3, 1) = +0.000000000000000;
            R0(3, 2) = +2.108756394574483;
            R0(3, 3) = +2.220980032989806;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 2;
            R0_block_sizes[1] = 2;
            break;

        // 5-stage 8th-order Lobatto IIIC
        case Type::LobIIIC8:
            is_imex = false;
            s = 5;
            s_eff = 3;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.050000000000000;
            A0(0, 1) = -0.116666666666667;
            A0(0, 2) = +0.133333333333333;
            A0(0, 3) = -0.116666666666667;
            A0(0, 4) = +0.050000000000000;
            A0(1, 0) = +0.050000000000000;
            A0(1, 1) = +0.161111111111111;
            A0(1, 2) = -0.069011541029643;
            A0(1, 3) = +0.052002165993115;
            A0(1, 4) = -0.021428571428571;
            A0(2, 0) = +0.050000000000000;
            A0(2, 1) = +0.281309183323043;
            A0(2, 2) = +0.202777777777778;
            A0(2, 3) = -0.052836961100821;
            A0(2, 4) = +0.018750000000000;
            A0(3, 0) = +0.050000000000000;
            A0(3, 1) = +0.270220056229107;
            A0(3, 2) = +0.367424239442342;
            A0(3, 3) = +0.161111111111111;
            A0(3, 4) = -0.021428571428571;
            A0(4, 0) = +0.050000000000000;
            A0(4, 1) = +0.272222222222222;
            A0(4, 2) = +0.355555555555556;
            A0(4, 3) = +0.272222222222222;
            A0(4, 4) = +0.050000000000000;
            /* --- inv(A) --- */
            invA0(0, 0) = +10.000000000000002;
            invA0(0, 1) = +13.513004977448478;
            invA0(0, 2) = -5.333333333333334;
            invA0(0, 3) = +2.820328355884856;
            invA0(0, 4) = -1.000000000000001;
            invA0(1, 0) = -2.481980506061966;
            invA0(1, 1) = +0.000000000000000;
            invA0(1, 2) = +3.491486243775877;
            invA0(1, 3) = -1.527525231651947;
            invA0(1, 4) = +0.518019493938034;
            invA0(2, 0) = +0.750000000000000;
            invA0(2, 1) = -2.673169155390907;
            invA0(2, 2) = +0.000000000000001;
            invA0(2, 3) = +2.673169155390906;
            invA0(2, 4) = -0.750000000000000;
            invA0(3, 0) = -0.518019493938035;
            invA0(3, 1) = +1.527525231651947;
            invA0(3, 2) = -3.491486243775877;
            invA0(3, 3) = -0.000000000000001;
            invA0(3, 4) = +2.481980506061966;
            invA0(4, 0) = +0.999999999999998;
            invA0(4, 1) = -2.820328355884851;
            invA0(4, 2) = +5.333333333333331;
            invA0(4, 3) = -13.513004977448482;
            invA0(4, 4) = +10.000000000000004;
            /* --- b --- */
            b0(0) = +0.050000000000000;
            b0(1) = +0.272222222222222;
            b0(2) = +0.355555555555556;
            b0(3) = +0.272222222222222;
            b0(4) = +0.050000000000000;
            /* --- c --- */
            c0(0) = +0.000000000000000;
            c0(1) = +0.172673164646011;
            c0(2) = +0.500000000000000;
            c0(3) = +0.827326835353989;
            c0(4) = +1.000000000000000;
            /* --- d --- */
            d0(0) = -0.000000000000000;
            d0(1) = -0.000000000000000;
            d0(2) = +0.000000000000000;
            d0(3) = -0.000000000000000;
            d0(4) = +1.000000000000000;
            /* --- Q --- */
            Q0(0, 0) = -0.571732147184072;
            Q0(0, 1) = -0.217020758096524;
            Q0(0, 2) = -0.319217568516840;
            Q0(0, 3) = +0.640115515555092;
            Q0(0, 4) = -0.338196116369386;
            Q0(1, 0) = +0.261426736644018;
            Q0(1, 1) = +0.145745772324781;
            Q0(1, 2) = +0.157982461885322;
            Q0(1, 3) = -0.130646891000507;
            Q0(1, 4) = -0.931872932768018;
            Q0(2, 0) = -0.124973465898830;
            Q0(2, 1) = -0.282976730324878;
            Q0(2, 2) = -0.710816244885500;
            Q0(2, 3) = -0.621569830676823;
            Q0(2, 4) = -0.112681029970164;
            Q0(3, 0) = -0.765238254685085;
            Q0(3, 1) = +0.325571308560971;
            Q0(3, 2) = +0.371091608018041;
            Q0(3, 3) = -0.410895349437400;
            Q0(3, 4) = -0.043240801076333;
            Q0(4, 0) = +0.059753323185469;
            Q0(4, 1) = +0.863474176534077;
            Q0(4, 2) = -0.479763452289052;
            Q0(4, 3) = +0.134162118426338;
            Q0(4, 4) = +0.051666649341120;
            /* --- R --- */
            R0(0, 0) = +2.664731518065850;
            R0(0, 1) = -3.395839948655504;
            R0(0, 2) = -5.334381337873263;
            R0(0, 3) = -5.795630734591422;
            R0(0, 4) = +9.491995902665616;
            R0(1, 0) = +10.195334979321485;
            R0(1, 1) = +2.664731518065850;
            R0(1, 2) = -13.338533879421322;
            R0(1, 3) = +2.816806937357789;
            R0(1, 4) = +5.035627803894975;
            R0(2, 0) = +0.000000000000000;
            R0(2, 1) = +0.000000000000000;
            R0(2, 2) = +5.277122810196474;
            R0(2, 3) = -3.815356369998933;
            R0(2, 4) = +1.959142180222903;
            R0(3, 0) = +0.000000000000000;
            R0(3, 1) = +0.000000000000000;
            R0(3, 2) = +0.000000000000000;
            R0(3, 3) = +4.696707076835921;
            R0(3, 4) = -10.644178015207251;
            R0(4, 0) = +0.000000000000000;
            R0(4, 1) = +0.000000000000000;
            R0(4, 2) = +0.000000000000000;
            R0(4, 3) = +0.795001566220291;
            R0(4, 4) = +4.696707076835921;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 2;
            R0_block_sizes[1] = 1;
            R0_block_sizes[2] = 2;
            break;

        // 2-stage 3rd-order Radau IIA
        case Type::IMEXRadauIIA3:
            is_imex = true;
            s = 2 ;
            s_eff = 1 ;
            alpha = 2.0 ;
            order = 3 ;
            SizeData();
            /* --- A --- */
            A0( 0 , 0 ) = 0.0 ;
            A0( 0 , 1 ) = 0.0 ;
            A0( 0 , 2 ) = 0.0 ;
            A0( 1 , 0 ) = 0.0 ;
            A0( 1 , 1 ) = 0.8333333333333334 ;
            A0( 1 , 2 ) = -0.16666666666666666 ;
            A0( 2 , 0 ) = 0.0 ;
            A0( 2 , 1 ) = 1.5 ;
            A0( 2 , 2 ) = 0.5 ;
            /* --- exp A --- */
            if (star_method) {
                order += 1;
                expA0( 0 , 0 ) = 0.0 ;
                expA0( 0 , 1 ) = 0.0 ;
                expA0( 0 , 2 ) = 0.0 ;
                expA0( 1 , 0 ) = 0.2962962962962963 ;
                expA0( 1 , 1 ) = -0.6111111111111112 ;
                expA0( 1 , 2 ) = 0.9814814814814815 ;
                expA0( 2 , 0 ) = 4.0 ;
                expA0( 2 , 1 ) = -7.5 ;
                expA0( 2 , 2 ) = 5.5 ;
            }
            else {
                expA0( 0 , 0 ) = 0.0 ;
                expA0( 0 , 1 ) = 0.0 ;
                expA0( 0 , 2 ) = 0.0 ;
                expA0( 1 , 0 ) = 0.0 ;
                expA0( 1 , 1 ) = -0.16666666666666666 ;
                expA0( 1 , 2 ) = 0.8333333333333333 ;
                expA0( 2 , 0 ) = 0.0 ;
                expA0( 2 , 1 ) = -1.5 ;
                expA0( 2 , 2 ) = 3.50 ;
            }
            /* --- expA_it --- */
            expA0_it( 0 , 0 ) = 0.0 ;
            expA0_it( 0 , 1 ) = 0.0 ;
            expA0_it( 0 , 2 ) = 0.0 ;
            expA0_it( 1 , 0 ) = 0.0 ;
            expA0_it( 1 , 1 ) = 0.8333333333333334 ;
            expA0_it( 1 , 2 ) = -0.16666666666666666 ;
            expA0_it( 2 , 0 ) = 0.0 ;
            expA0_it( 2 , 1 ) = 1.5 ;
            expA0_it( 2 , 2 ) = 0.5 ;
            /* --- inv(A) --- */
            invA0( 0 , 0 ) = 0.75 ;
            invA0( 0 , 1 ) = 0.25 ;
            invA0( 1 , 0 ) = -2.25 ;
            invA0( 1 , 1 ) = 1.25 ;
            /* --- Q --- */
            Q0( 0 , 0 ) = 0.992507556682903 ;
            Q0( 0 , 1 ) = 0.12218326369570452 ;
            Q0( 1 , 0 ) = -0.12218326369570452 ;
            Q0( 1 , 1 ) = 0.992507556682903 ;
            /* --- R --- */
            R0( 0 , 0 ) = 1.0 ;
            R0( 0 , 1 ) = 0.2192235935955848 ;
            R0( 1 , 0 ) = -2.2807764064044154 ;
            R0( 1 , 1 ) = 1.0 ;
            /* --- z --- */
            z0( 0 ) = 0.0 ;
            z0( 1 ) = 0.6666666666666666 ;
            z0( 2 ) = 2.0 ;        
            /* --- R block sizes --- */
            R0_block_sizes[ 0 ] = 2.0 ;
            break;

        // 3-stage 4th-order Radau IIA
        case Type::IMEXRadauIIA4:
            is_imex = true;
            s = 3 ;
            s_eff = 2 ;
            alpha = 2 ;
            order = 4 ;
            SizeData();
            /* --- A --- */
            A0( 0 , 0 ) = 0.0 ;
            A0( 0 , 1 ) = 0.0 ;
            A0( 0 , 2 ) = 0.0 ;
            A0( 0 , 3 ) = 0.0 ;
            A0( 1 , 0 ) = 0.0 ;
            A0( 1 , 1 ) = 0.3936309544473202 ;
            A0( 1 , 2 ) = -0.13107085170039648 ;
            A0( 1 , 3 ) = 0.047541948696440246 ;
            A0( 2 , 0 ) = 0.0 ;
            A0( 2 , 1 ) = 0.7888486294781741 ;
            A0( 2 , 2 ) = 0.5841468233304582 ;
            A0( 2 , 3 ) = -0.0830975042519963 ;
            A0( 3 , 0 ) = 0.0 ;
            A0( 3 , 1 ) = 0.7528061254009342 ;
            A0( 3 , 2 ) = 1.0249716523768442 ;
            A0( 3 , 3 ) = 0.22222222222222165 ;
            /* --- exp A --- */
            if (star_method) {
                order += 1;
                expA0( 0 , 0 ) = 0.0 ;
                expA0( 0 , 1 ) = 0.0 ;
                expA0( 0 , 2 ) = 0.0 ;
                expA0( 0 , 3 ) = 0.0 ;
                expA0( 1 , 0 ) = -0.10483269811986462 ;
                expA0( 1 , 1 ) = 0.18996165838584106 ;
                expA0( 1 , 2 ) = -0.2245199283702215 ;
                expA0( 1 , 3 ) = 0.44949301954760895 ;
                expA0( 2 , 0 ) = -4.2591673018801375 ;
                expA0( 2 , 1 ) = 7.424964372814666 ;
                expA0( 2 , 2 ) = -6.845517213941396 ;
                expA0( 2 , 3 ) = 4.969618091563503 ;
                expA0( 3 , 0 ) = -16.00000000000001 ;
                expA0( 3 , 1 ) = 27.397533467493776 ;
                expA0( 3 , 2 ) = -22.953089023049323 ;
                expA0( 3 , 3 ) = 13.555555555555557 ;
            }
            else {
                expA0( 0 , 0 ) = 0.0 ;
                expA0( 0 , 1 ) = 0.0 ;
                expA0( 0 , 2 ) = 0.0 ;
                expA0( 0 , 3 ) = 0.0 ;
                expA0( 1 , 0 ) = 0.0 ;
                expA0( 1 , 1 ) = 0.026624116302772838 ;
                expA0( 1 , 2 ) = -0.1310708517003964 ;
                expA0( 1 , 3 ) = 0.41454878684098745 ;
                expA0( 2 , 0 ) = 0.0 ;
                expA0( 2 , 1 ) = 0.7888486294781738 ;
                expA0( 2 , 2 ) = -3.048846338524995 ;
                expA0( 2 , 3 ) = 3.549895657603457 ;
                expA0( 3 , 0 ) = 0.0 ;
                expA0( 3 , 1 ) = 2.468282191895014 ;
                expA0( 3 , 2 ) = -8.690504414117237 ;
                expA0( 3 , 3 ) = 8.222222222222223 ;
            }
            /* --- expA_it --- */
            expA0_it( 0 , 0 ) = 0.0 ;
            expA0_it( 0 , 1 ) = 0.0 ;
            expA0_it( 0 , 2 ) = 0.0 ;
            expA0_it( 0 , 3 ) = 0.0 ;
            expA0_it( 1 , 0 ) = 0.0 ;
            expA0_it( 1 , 1 ) = 0.3936309544473202 ;
            expA0_it( 1 , 2 ) = -0.13107085170039648 ;
            expA0_it( 1 , 3 ) = 0.047541948696440246 ;
            expA0_it( 2 , 0 ) = 0.0 ;
            expA0_it( 2 , 1 ) = 0.7888486294781741 ;
            expA0_it( 2 , 2 ) = 0.5841468233304582 ;
            expA0_it( 2 , 3 ) = -0.0830975042519963 ;
            expA0_it( 3 , 0 ) = 0.0 ;
            expA0_it( 3 , 1 ) = 0.7528061254009342 ;
            expA0_it( 3 , 2 ) = 1.0249716523768442 ;
            expA0_it( 3 , 3 ) = 0.22222222222222165 ;
            /* --- inv(A) --- */
            invA0( 0 , 0 ) = 1.6123724356957994 ;
            invA0( 0 , 1 ) = 0.583920042345202 ;
            invA0( 0 , 2 ) = -0.12659863237109054 ;
            invA0( 1 , 0 ) = -1.7839200423452022 ;
            invA0( 1 , 1 ) = 0.3876275643042033 ;
            invA0( 1 , 2 ) = 0.5265986323710913 ;
            invA0( 2 , 0 ) = 2.765986323710903 ;
            invA0( 2 , 1 ) = -3.7659863237109024 ;
            invA0( 2 , 2 ) = 2.5 ;
            /* --- Q --- */
            Q0( 0 , 0 ) = 0.13866510875190746 ;
            Q0( 0 , 1 ) = 0.046278149309489106 ;
            Q0( 0 , 2 ) = 0.9892574591638471 ;
            Q0( 1 , 0 ) = -0.2296412423517415 ;
            Q0( 1 , 1 ) = -0.9701788865518328 ;
            Q0( 1 , 2 ) = 0.07757466016809347 ;
            Q0( 2 , 0 ) = -0.9633467119505681 ;
            Q0( 2 , 1 ) = 0.23793121061671393 ;
            Q0( 2 , 2 ) = 0.12390258911134387 ;
            /* --- R --- */
            R0( 0 , 0 ) = 1.3405414368138748 ;
            R0( 0 , 1 ) = -4.211937899042693 ;
            R0( 0 , 2 ) = -2.0442889065298107 ;
            R0( 1 , 0 ) = 0.5523065999260999 ;
            R0( 1 , 1 ) = 1.3405414368138748 ;
            R0( 1 , 2 ) = 2.35007600317197 ;
            R0( 2 , 0 ) = 0.0 ;
            R0( 2 , 1 ) = 0.0 ;
            R0( 2 , 2 ) = 1.818917126372251 ;
            /* --- z --- */
            z0( 0 ) = 0.0 ;
            z0( 1 ) = 1-0.689897948556636 ;
            z0( 2 ) = 1.289897948556636 ;
            z0( 3 ) = 2.0 ;
            /* --- R block sizes --- */
            R0_block_sizes[ 0 ] = 2.0 ;
            R0_block_sizes[ 1 ] = 1.0 ;
            break;
        // 4-stage 5th-order Radau IIA
        case Type::IMEXRadauIIA5:
            is_imex = true;
            s = 4 ;
            s_eff = 2 ;
            alpha = 2 ;
            order = 5 ;
            SizeData();
            /* --- A --- */
            A0( 0 , 0 ) = 0.0 ;
            A0( 0 , 1 ) = 0.0 ;
            A0( 0 , 2 ) = -0.0 ;
            A0( 0 , 3 ) = 0.0 ;
            A0( 0 , 4 ) = 0.0 ;
            A0( 1 , 0 ) = 0.0 ;
            A0( 1 , 1 ) = 0.22599895864631253 ;
            A0( 1 , 2 ) = -0.08061844144704443 ;
            A0( 1 , 3 ) = 0.05160475484067269 ;
            A0( 1 , 4 ) = -0.019809353014532807 ;
            A0( 2 , 0 ) = 0.0 ;
            A0( 2 , 1 ) = 0.4687679914948005 ;
            A0( 2 , 2 ) = 0.413785147870717 ;
            A0( 2 , 3 ) = -0.09571425609708101 ;
            A0( 2 , 4 ) = 0.03209484561303241 ;
            A0( 3 , 0 ) = 0.0 ;
            A0( 3 , 1 ) = 0.43336356924650044 ;
            A0( 3 , 2 ) = 0.8122465277347463 ;
            A0( 3 , 3 ) = 0.37807303634011313 ;
            A0( 3 , 4 ) = -0.04836420979966604 ;
            A0( 4 , 0 ) = 0.0 ;
            A0( 4 , 1 ) = 0.4409244223535369 ;
            A0( 4 , 2 ) = 0.7763869376863433 ;
            A0( 4 , 3 ) = 0.6576886399601202 ;
            A0( 4 , 4 ) = 0.12499999999999982 ;
            /* --- exp A --- */
            if (star_method) {
                order += 1;
                expA0( 0 , 0 ) = 0.0 ;
                expA0( 0 , 1 ) = -0.0 ;
                expA0( 0 , 2 ) = 0.0 ;
                expA0( 0 , 3 ) = 0.0 ;
                expA0( 0 , 4 ) = 0.0 ;
                expA0( 1 , 0 ) = 0.047219800436129555 ;
                expA0( 1 , 1 ) = -0.08127513390077432 ;
                expA0( 1 , 2 ) = 0.07486765592616924 ;
                expA0( 1 , 3 ) = -0.11869765044598872 ;
                expA0( 1 , 4 ) = 0.2550612470098721 ;
                expA0( 2 , 0 ) = 3.0481766815607942 ;
                expA0( 2 , 1 ) = -5.15993259077104 ;
                expA0( 2 , 2 ) = 4.334539551675012 ;
                expA0( 2 , 3 ) = -5.051811325599685 ;
                expA0( 2 , 4 ) = 3.6479614120163872 ;
                expA0( 3 , 0 ) = 28.046627674604494 ;
                expA0( 3 , 1 ) = -46.84583333910275 ;
                expA0( 3 , 2 ) = 36.77781291429983 ;
                expA0( 3 , 3 ) = -36.213978703488515 ;
                expA0( 3 , 4 ) = 19.81069037720863 ;
                expA0( 4 , 0 ) = 68.00000000000006 ;
                expA0( 4 , 1 ) = -112.94958989666215 ;
                expA0( 4 , 2 ) = 86.34154857357431 ;
                expA0( 4 , 3 ) = -80.26695867691221 ;
                expA0( 4 , 4 ) = 40.87499999999999 ;
            }
            else {
                expA0( 0 , 0 ) = 0.0 ;
                expA0( 0 , 1 ) = 0.0 ;
                expA0( 0 , 2 ) = -0.0 ;
                expA0( 0 , 3 ) = 0.0 ;
                expA0( 0 , 4 ) = 0.0 ;
                expA0( 1 , 0 ) = 0.0 ;
                expA0( 1 , 1 ) = -0.006784121370153905 ;
                expA0( 1 , 2 ) = 0.028890841411446995 ;
                expA0( 1 , 3 ) = -0.08818709791672501 ;
                expA0( 1 , 4 ) = 0.24325629690083977 ;
                expA0( 2 , 0 ) = 0.0 ;
                expA0( 2 , 1 ) = -0.35131914292688093 ;
                expA0( 2 , 2 ) = 1.3666012588017453 ;
                expA0( 2 , 3 ) = -3.0822656286195853 ;
                expA0( 2 , 4 ) = 2.8859172416261902 ;
                expA0( 3 , 0 ) = 0.0 ;
                expA0( 3 , 1 ) = -2.6012225136815026 ;
                expA0( 3 , 2 ) = 9.469467973220128 ;
                expA0( 3 , 3 ) = -18.091959994574438 ;
                expA0( 3 , 4 ) = 12.799033458557506 ;
                expA0( 4 , 0 ) = 0.0 ;
                expA0( 4 , 1 ) = -5.6770303920167215 ;
                expA0( 4 , 2 ) = 20.13154009990472 ;
                expA0( 4 , 3 ) = -36.329509707887986 ;
                expA0( 4 , 4 ) = 23.87499999999999 ;
            }
            /* --- expA_it --- */
            expA0_it( 0 , 0 ) = 0.0 ;
            expA0_it( 0 , 1 ) = 0.0 ;
            expA0_it( 0 , 2 ) = -0.0 ;
            expA0_it( 0 , 3 ) = 0.0 ;
            expA0_it( 0 , 4 ) = 0.0 ;
            expA0_it( 1 , 0 ) = 0.0 ;
            expA0_it( 1 , 1 ) = 0.22599895864631253 ;
            expA0_it( 1 , 2 ) = -0.08061844144704443 ;
            expA0_it( 1 , 3 ) = 0.05160475484067269 ;
            expA0_it( 1 , 4 ) = -0.019809353014532807 ;
            expA0_it( 2 , 0 ) = 0.0 ;
            expA0_it( 2 , 1 ) = 0.4687679914948005 ;
            expA0_it( 2 , 2 ) = 0.413785147870717 ;
            expA0_it( 2 , 3 ) = -0.09571425609708101 ;
            expA0_it( 2 , 4 ) = 0.03209484561303241 ;
            expA0_it( 3 , 0 ) = 0.0 ;
            expA0_it( 3 , 1 ) = 0.43336356924650044 ;
            expA0_it( 3 , 2 ) = 0.8122465277347463 ;
            expA0_it( 3 , 3 ) = 0.37807303634011313 ;
            expA0_it( 3 , 4 ) = -0.04836420979966604 ;
            expA0_it( 4 , 0 ) = 0.0 ;
            expA0_it( 4 , 1 ) = 0.4409244223535369 ;
            expA0_it( 4 , 2 ) = 0.7763869376863433 ;
            expA0_it( 4 , 3 ) = 0.6576886399601202 ;
            expA0_it( 4 , 4 ) = 0.12499999999999982 ;
            /* --- inv(A) --- */
            invA0( 0 , 0 ) = 2.8220539379750416 ;
            invA0( 0 , 1 ) = 0.9617536385273565 ;
            invA0( 0 , 2 ) = -0.2929507410519071 ;
            invA0( 0 , 3 ) = 0.08693917628712232 ;
            invA0( 1 , 0 ) = -2.524607319195707 ;
            invA0( 1 , 1 ) = 0.6105500144473492 ;
            invA0( 1 , 2 ) = 0.8773404943804166 ;
            invA0( 1 , 3 ) = -0.21739573060629003 ;
            invA0( 2 , 0 ) = 1.746233079312721 ;
            invA0( 2 , 1 ) = -1.99225894789125 ;
            invA0( 2 , 2 ) = 0.3173960475776096 ;
            invA0( 2 , 3 ) = 0.9110687992171267 ;
            invA0( 3 , 0 ) = -3.4617441282227426 ;
            invA0( 3 , 1 ) = 3.297618834814078 ;
            invA0( 3 , 2 ) = -6.085874706591345 ;
            invA0( 3 , 3 ) = 4.249999999999998 ;
            /* --- Q --- */
            Q0( 0 , 0 ) = 0.05457029299477517 ;
            Q0( 0 , 1 ) = 0.12568329449502288 ;
            Q0( 0 , 2 ) = 0.956633137728588 ;
            Q0( 0 , 3 ) = -0.25705803314990733 ;
            Q0( 1 , 0 ) = -0.17766711973873603 ;
            Q0( 1 , 1 ) = -0.1269912622387495 ;
            Q0( 1 , 2 ) = 0.2781655592008429 ;
            Q0( 1 , 3 ) = 0.9353777501914572 ;
            Q0( 2 , 0 ) = 0.3212541312291417 ;
            Q0( 2 , 1 ) = -0.9391937539441212 ;
            Q0( 2 , 2 ) = 0.08074733999507169 ;
            Q0( 2 , 3 ) = -0.0905027226346195 ;
            Q0( 3 , 0 ) = 0.9285753931988595 ;
            Q0( 3 , 1 ) = 0.29324396217524307 ;
            Q0( 3 , 2 ) = -0.03093264550212032 ;
            Q0( 3 , 3 ) = 0.22538608926814732 ;
            /* --- R --- */
            R0( 0 , 0 ) = 1.6064034484357672 ;
            R0( 0 , 1 ) = 6.070785522887777 ;
            R0( 0 , 2 ) = -1.8982931086944816 ;
            R0( 0 , 3 ) = 4.223380722325066 ;
            R0( 1 , 0 ) = -0.9381966946373947 ;
            R0( 1 , 1 ) = 1.6064034484357672 ;
            R0( 1 , 2 ) = -1.2859322633832986 ;
            R0( 1 , 3 ) = 3.502841910999233 ;
            R0( 2 , 0 ) = 0.0 ;
            R0( 2 , 1 ) = 0.0 ;
            R0( 2 , 2 ) = 2.3935965515642343 ;
            R0( 2 , 3 ) = 0.17232565003075662 ;
            R0( 3 , 0 ) = 0.0 ;
            R0( 3 , 1 ) = 0.0 ;
            R0( 3 , 2 ) = -3.5644466118133207 ;
            R0( 3 , 3 ) = 2.3935965515642343 ;
            /* --- z --- */
            z0( 0 ) = 0.0 ;
            z0( 1 ) = 1-0.822824080974592 ;
            z0( 2 ) = 1-0.181066271118531 ;
            z0( 3 ) = 1.575318923521694 ;
            z0( 4 ) = 2.0 ;
            /* --- R block sizes --- */
            R0_block_sizes[ 0 ] = 2.0 ;
            R0_block_sizes[ 1 ] = 2.0 ;
            break;
        // 5-stage 6th-order Radau IIA
        case Type::IMEXRadauIIA6:
            is_imex = true;
            s = 5 ;
            s_eff = 3 ;
            alpha = 2 ;
            order = 6 ;
            SizeData();
            /* --- A --- */
            A0( 0 , 0 ) = 0.0 ;
            A0( 0 , 1 ) = 0.0 ;
            A0( 0 , 2 ) = 0.0 ;
            A0( 0 , 3 ) = 0.0 ;
            A0( 0 , 4 ) = 0.0 ;
            A0( 0 , 5 ) = 0.0 ;
            A0( 1 , 0 ) = 0.0 ;
            A0( 1 , 1 ) = 0.145997728635806 ;
            A0( 1 , 2 ) = -0.053470662215890793 ;
            A0( 1 , 3 ) = 0.03735385952796852 ;
            A0( 1 , 4 ) = -0.025758212186612735 ;
            A0( 1 , 5 ) = 0.010085678467763976 ;
            A0( 2 , 0 ) = 0.0 ;
            A0( 2 , 1 ) = 0.3075504629583645 ;
            A0( 2 , 2 ) = 0.2924297356949882 ;
            A0( 2 , 3 ) = -0.07288913781025681 ;
            A0( 2 , 4 ) = 0.04246612623860985 ;
            A0( 2 , 5 ) = -0.015871159805457714 ;
            A0( 3 , 0 ) = 0.0 ;
            A0( 3 , 1 ) = 0.2801260913696192 ;
            A0( 3 , 2 ) = 0.597934258982568 ;
            A0( 3 , 3 ) = 0.3351701402704973 ;
            A0( 3 , 4 ) = -0.06793820337323504 ;
            A0( 3 , 5 ) = 0.02188857748838429 ;
            A0( 4 , 0 ) = 0.0 ;
            A0( 4 , 1 ) = 0.28978861621906915 ;
            A0( 4 , 2 ) = 0.5530001375203198 ;
            A0( 4 , 3 ) = 0.6515958458208408 ;
            A0( 4 , 4 ) = 0.2575135065098202 ;
            A0( 4 , 5 ) = -0.03141783475761097 ;
            A0( 5 , 0 ) = 0.0 ;
            A0( 5 , 1 ) = 0.2874271215824512 ;
            A0( 5 , 2 ) = 0.562712030298925 ;
            A0( 5 , 3 ) = 0.6236530459514823 ;
            A0( 5 , 4 ) = 0.44620780216714134 ;
            A0( 5 , 5 ) = 0.07999999999999996 ;
            /* --- exp A --- */
            if (star_method) {
                order += 1;
                expA0( 0 , 0 ) = 0.0 ;
                expA0( 0 , 1 ) = 0.0 ;
                expA0( 0 , 2 ) = -0.0 ;
                expA0( 0 , 3 ) = 0.0 ;
                expA0( 0 , 4 ) = 0.0 ;
                expA0( 0 , 5 ) = 0.0 ;
                expA0( 1 , 0 ) = -0.024938835224603953 ;
                expA0( 1 , 1 ) = 0.04186243193304367 ;
                expA0( 1 , 2 ) = -0.03426586505493761 ;
                expA0( 1 , 3 ) = 0.0415652999419458 ;
                expA0( 1 , 4 ) = -0.0738592178586334 ;
                expA0( 1 , 5 ) = 0.16384457849222062 ;
                expA0( 2 , 0 ) = -2.0497334883340192 ;
                expA0( 2 , 1 ) = 3.4131791826822258 ;
                expA0( 2 , 2 ) = -2.6817669086700713 ;
                expA0( 2 , 3 ) = 2.9063822413470355 ;
                expA0( 2 , 4 ) = -3.6926752486552608 ;
                expA0( 2 , 5 ) = 2.658300248906338 ;
                expA0( 3 , 0 ) = -31.07325886752193 ;
                expA0( 3 , 1 ) = 51.310783931151924 ;
                expA0( 3 , 2 ) = -38.70690385646687 ;
                expA0( 3 , 3 ) = 38.05776206425789 ;
                expA0( 3 , 4 ) = -39.385342249555244 ;
                expA0( 3 , 5 ) = 20.96413984287206 ;
                expA0( 4 , 0 ) = -156.44539606175033 ;
                expA0( 4 , 1 ) = 256.92118012762575 ;
                expA0( 4 , 2 ) = -188.89844912976812 ;
                expA0( 4 , 3 ) = 175.71899860943256 ;
                expA0( 4 , 4 ) = -165.37193166159503 ;
                expA0( 4 , 5 ) = 79.79607838736763 ;
                expA0( 5 , 0 ) = -304.0000000000007 ;
                expA0( 5 , 1 ) = 498.1189532441582 ;
                expA0( 5 , 2 ) = -362.4796701852426 ;
                expA0( 5 , 3 ) = 330.14752220271197 ;
                expA0( 5 , 4 ) = -300.666805261627 ;
                expA0( 5 , 5 ) = 140.88000000000008 ;
            }
            else {
                expA0( 0 , 0 ) = 0.0 ;
                expA0( 0 , 1 ) = 0.0 ;
                expA0( 0 , 2 ) = 0.0 ;
                expA0( 0 , 3 ) = 0.0 ;
                expA0( 0 , 4 ) = 0.0 ;
                expA0( 0 , 5 ) = 0.0 ;
                expA0( 1 , 0 ) = 0.0 ;
                expA0( 1 , 1 ) = 0.002299266711286902 ;
                expA0( 1 , 2 ) = -0.009124579321272901 ;
                expA0( 1 , 3 ) = 0.023335638201459778 ;
                expA0( 1 , 4 ) = -0.0611587448097384 ;
                expA0( 1 , 5 ) = 0.15885681144729974 ;
                expA0( 2 , 0 ) = 0.0 ;
                expA0( 2 , 1 ) = 0.1614657835126635 ;
                expA0( 2 , 2 ) = -0.6153939267745175 ;
                expA0( 2 , 3 ) = 1.4080785790488428 ;
                expA0( 2 , 4 ) = -2.648817959750275 ;
                expA0( 2 , 5 ) = 2.2483535512395343 ;
                expA0( 3 , 0 ) = 0.0 ;
                expA0( 3 , 1 ) = 2.015920579180351 ;
                expA0( 3 , 2 ) = -7.381395955690257 ;
                expA0( 3 , 3 ) = 15.343990822422667 ;
                expA0( 3 , 4 ) = -23.56082265054261 ;
                expA0( 3 , 5 ) = 14.749488069367684 ;
                expA0( 4 , 0 ) = 0.0 ;
                expA0( 4 , 1 ) = 8.734967867486368 ;
                expA0( 4 , 2 ) = -31.183047881823065 ;
                expA0( 4 , 3 ) = 61.36134616187989 ;
                expA0( 4 , 4 ) = -85.69978505124837 ;
                expA0( 4 , 5 ) = 48.50699917501762 ;
                expA0( 5 , 0 ) = 0.0 ;
                expA0( 5 , 1 ) = 15.850951587514189 ;
                expA0( 5 , 2 ) = -56.0118342097376 ;
                expA0( 5 , 3 ) = 107.93116289013648 ;
                expA0( 5 , 4 ) = -145.8502802679131 ;
                expA0( 5 , 5 ) = 80.08000000000003 ;
            }

            /* --- expA_it --- */
            expA0_it( 0 , 0 ) = 0.0 ;
            expA0_it( 0 , 1 ) = 0.0 ;
            expA0_it( 0 , 2 ) = 0.0 ;
            expA0_it( 0 , 3 ) = 0.0 ;
            expA0_it( 0 , 4 ) = 0.0 ;
            expA0_it( 0 , 5 ) = 0.0 ;
            expA0_it( 1 , 0 ) = 0.0 ;
            expA0_it( 1 , 1 ) = 0.145997728635806 ;
            expA0_it( 1 , 2 ) = -0.053470662215890793 ;
            expA0_it( 1 , 3 ) = 0.03735385952796852 ;
            expA0_it( 1 , 4 ) = -0.025758212186612735 ;
            expA0_it( 1 , 5 ) = 0.010085678467763976 ;
            expA0_it( 2 , 0 ) = 0.0 ;
            expA0_it( 2 , 1 ) = 0.3075504629583645 ;
            expA0_it( 2 , 2 ) = 0.2924297356949882 ;
            expA0_it( 2 , 3 ) = -0.07288913781025681 ;
            expA0_it( 2 , 4 ) = 0.04246612623860985 ;
            expA0_it( 2 , 5 ) = -0.015871159805457714 ;
            expA0_it( 3 , 0 ) = 0.0 ;
            expA0_it( 3 , 1 ) = 0.2801260913696192 ;
            expA0_it( 3 , 2 ) = 0.597934258982568 ;
            expA0_it( 3 , 3 ) = 0.3351701402704973 ;
            expA0_it( 3 , 4 ) = -0.06793820337323504 ;
            expA0_it( 3 , 5 ) = 0.02188857748838429 ;
            expA0_it( 4 , 0 ) = 0.0 ;
            expA0_it( 4 , 1 ) = 0.28978861621906915 ;
            expA0_it( 4 , 2 ) = 0.5530001375203198 ;
            expA0_it( 4 , 3 ) = 0.6515958458208408 ;
            expA0_it( 4 , 4 ) = 0.2575135065098202 ;
            expA0_it( 4 , 5 ) = -0.03141783475761097 ;
            expA0_it( 5 , 0 ) = 0.0 ;
            expA0_it( 5 , 1 ) = 0.2874271215824512 ;
            expA0_it( 5 , 2 ) = 0.562712030298925 ;
            expA0_it( 5 , 3 ) = 0.6236530459514823 ;
            expA0_it( 5 , 4 ) = 0.44620780216714134 ;
            expA0_it( 5 , 5 ) = 0.07999999999999996 ;
            /* --- inv(A) --- */
            invA0( 0 , 0 ) = 4.377961988969214 ;
            invA0( 0 , 1 ) = 1.4459713076900529 ;
            invA0( 0 , 2 ) = -0.43759319810013186 ;
            invA0( 0 , 3 ) = 0.19985260396998258 ;
            invA0( 0 , 4 ) = -0.06685308192460772 ;
            invA0( 1 , 0 ) = -3.580690360072697 ;
            invA0( 1 , 1 ) = 0.9030388620418195 ;
            invA0( 1 , 2 ) = 1.1818985880343047 ;
            invA0( 1 , 3 ) = -0.4329503901415677 ;
            invA0( 1 , 4 ) = 0.1371690388875972 ;
            invA0( 2 , 0 ) = 2.0610826231216937 ;
            invA0( 2 , 1 ) = -2.2480085629066995 ;
            invA0( 2 , 2 ) = 0.42838262269859095 ;
            invA0( 2 , 3 ) = 1.259160474605531 ;
            invA0( 2 , 4 ) = -0.3285313785671793 ;
            invA0( 3 , 0 ) = -1.9393316098620117 ;
            invA0( 3 , 1 ) = 1.6965759590324785 ;
            invA0( 3 , 2 ) = -2.5941704532035965 ;
            invA0( 3 , 3 ) = 0.29061652629041007 ;
            invA0( 3 , 4 ) = 1.4049918276398559 ;
            invA0( 4 , 0 ) = 4.2062121117971145 ;
            invA0( 4 , 1 ) = -3.4851280583282858 ;
            invA0( 4 , 2 ) = 4.388557102075214 ;
            invA0( 4 , 3 ) = -9.109641155544042 ;
            invA0( 4 , 4 ) = 6.499999999999998 ;
            /* --- Q --- */
            Q0( 0 , 0 ) = 0.10437601186510755 ;
            Q0( 0 , 1 ) = 0.0027532156993361444 ;
            Q0( 0 , 2 ) = 0.03463118326709646 ;
            Q0( 0 , 3 ) = 0.42357418712801015 ;
            Q0( 0 , 4 ) = 0.8991571926503251 ;
            Q0( 1 , 0 ) = -0.21606192229727592 ;
            Q0( 1 , 1 ) = -0.03221537578206395 ;
            Q0( 1 , 2 ) = -0.06458442826369405 ;
            Q0( 1 , 3 ) = -0.8699056147695395 ;
            Q0( 1 , 4 ) = 0.4374614135151904 ;
            Q0( 2 , 0 ) = 0.2913160484418208 ;
            Q0( 2 , 1 ) = 0.1925223198930282 ;
            Q0( 2 , 2 ) = 0.9252483635163872 ;
            Q0( 2 , 3 ) = -0.14827523362384973 ;
            Q0( 2 , 4 ) = -0.00019278819627801315 ;
            Q0( 3 , 0 ) = 0.8694415655327663 ;
            Q0( 3 , 1 ) = -0.40101786619435736 ;
            Q0( 3 , 2 ) = -0.22018315256222779 ;
            Q0( 3 , 3 ) = -0.1864509466211691 ;
            Q0( 3 , 4 ) = -0.0033851068945508723 ;
            Q0( 4 , 0 ) = -0.31879337810644237 ;
            Q0( 4 , 1 ) = -0.8950276066705789 ;
            Q0( 4 , 2 ) = 0.3001072773349188 ;
            Q0( 4 , 3 ) = 0.08425929615857551 ;
            Q0( 4 , 4 ) = -0.011504715315961601 ;
            /* --- R --- */
            R0( 0 , 0 ) = 1.8278471627317763 ;
            R0( 0 , 1 ) = -1.380748879449467 ;
            R0( 0 , 2 ) = -4.80370672224281 ;
            R0( 0 , 3 ) = -2.3412674257646495 ;
            R0( 0 , 4 ) = -0.25016758348632867 ;
            R0( 1 , 0 ) = 7.753128256225703 ;
            R0( 1 , 1 ) = 1.8278471627317763 ;
            R0( 1 , 2 ) = -6.606511724079506 ;
            R0( 1 , 3 ) = -4.434478387245237 ;
            R0( 1 , 4 ) = -1.3011176974410463 ;
            R0( 2 , 0 ) = 0.0 ;
            R0( 2 , 1 ) = 0.0 ;
            R0( 2 , 2 ) = 3.1433523758646174 ;
            R0( 2 , 3 ) = 4.821340639364689 ;
            R0( 2 , 4 ) = 2.033185899895442 ;
            R0( 3 , 0 ) = 0.0 ;
            R0( 3 , 1 ) = 0.0 ;
            R0( 3 , 2 ) = 0.0 ;
            R0( 3 , 3 ) = 2.8504766493359206 ;
            R0( 3 , 4 ) = 4.640191084730031 ;
            R0( 4 , 0 ) = 0.0 ;
            R0( 4 , 1 ) = 0.0 ;
            R0( 4 , 2 ) = 0.0 ;
            R0( 4 , 3 ) = -0.5552468118413603 ;
            R0( 4 , 4 ) = 2.8504766493359206 ;
            /* --- z --- */
            z0( 0 ) = 0.0 ;
            z0( 1 ) = 0.114208392229035 ;
            z0( 2 ) = 0.553686027276248 ;
            z0( 3 ) = 1.167180864737834 ;
            z0( 4 ) = 1.720480271312439 ;
            z0( 5 ) = 2.0 ;
            /* --- R block sizes --- */
            R0_block_sizes[ 0 ] = 2.0 ;
            R0_block_sizes[ 1 ] = 1.0 ;
            R0_block_sizes[ 2 ] = 2.0 ;
            break;
        default:
            mfem_error("RKData:: Invalid Runge Kutta type.\n");
    }    
}
