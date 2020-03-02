#include "IRK.hpp"

IRK::IRK(MPI_Comm globComm, int RK_ID) {
    
    m_RK_ID = RK_ID;
    
    SetButcherCoeffs();
    SetXCoeffs();
    
    
    for (int j = 0; j < m_s; j++) {
        for (int k = 0; k < m_s; k++) {
            std::cout << "x[" << j << "][" << k << "]=" << m_XCoeffs[j][k] << '\n';
        }
        std::cout << '\n';
    }
}

/* Set constants from Butcher tables and associated parameters 

TODO: Will need to have a more systematic approach for organising these...
*/
void IRK::SetButcherCoeffs() {
    
    m_s = m_RK_ID; // Fix when properly implement Butcher tables...
    // Resize Butcher arrays
    m_A0.resize(m_s, std::vector<double>(m_s, 0.0));
    m_b0.resize(m_s);
    m_c0.resize(m_s);
    
    m_invA0.resize(m_s, std::vector<double>(m_s, 0.0));
    m_b0tilde.resize(m_s);
    m_beta.resize(m_s);
    m_eta.resize(m_s);

    
    // Backward Euler
    if (m_RK_ID == 1) {
        m_A0[0][0]=+1.0000000000000000;

        m_b0[0]=+1.0000000000000000;

        m_c0[0]=+1.0000000000000000;

        m_invA0[0][0]=+1.0000000000000000;

        m_b0tilde[0]=+1.0000000000000000;

        m_beta[0]=+0.0000000000000000;

        m_eta[0]=+1.0000000000000000;
    
    
    // implicit 4th-order method, Hammer & Hollingsworth (A-stable)
    // note: coincides with s=2-stage, p=2s-order Gauss method
    } else if (m_RK_ID == 2) {
        m_A0[0][0]=+0.2500000000000000;
        m_A0[0][1]=-0.0386751345948129;
        m_A0[1][0]=+0.5386751345948129;
        m_A0[1][1]=+0.2500000000000000;

        m_b0[0]=+0.5000000000000000;
        m_b0[1]=+0.5000000000000000;

        m_c0[0]=+0.2113248654051871;
        m_c0[1]=+0.7886751345948129;
        
        m_invA0[0][0]=+3.0000000000000004;
        m_invA0[0][1]=+0.4641016151377544;
        m_invA0[1][0]=-6.4641016151377553;
        m_invA0[1][1]=+3.0000000000000004;

        m_b0tilde[0]=-1.7320508075688774;
        m_b0tilde[1]=+1.7320508075688774;

        m_beta[0]=-1.7320508075688772;
        m_beta[1]=+1.7320508075688772;

        m_eta[0]=+3.0000000000000000;
        m_eta[1]=+3.0000000000000000;
    
    /* A 6th-order Gauss--Legendre method */
    } else if (m_RK_ID == 3) {
        m_A0[0][0]=+0.1388888888888889;
        m_A0[0][1]=-0.0359766675249389;
        m_A0[0][2]=+0.0097894440153083;
        m_A0[1][0]=+0.3002631949808646;
        m_A0[1][1]=+0.2222222222222222;
        m_A0[1][2]=-0.0224854172030868;
        m_A0[2][0]=+0.8459956700754365;
        m_A0[2][1]=+0.4804211119693834;
        m_A0[2][2]=+0.1388888888888889;

        m_b0[0]=+0.2777777777777778;
        m_b0[1]=+0.4444444444444444;
        m_b0[2]=+0.2777777777777778;

        m_c0[0]=+0.1127016653792583;
        m_c0[1]=+0.5000000000000000;
        m_c0[2]=+0.8872983346207417;

        m_invA0[0][0]=+5.5235213928655451;
        m_invA0[0][1]=+1.2858512502378714;
        m_invA0[0][2]=-0.1811469716647615;
        m_invA0[1][0]=-8.0500844219791698;
        m_invA0[1][1]=+1.4593094304129137;
        m_invA0[1][2]=+0.8036570313986691;
        m_invA0[2][0]=-5.7992016418462988;
        m_invA0[2][1]=-12.8801350751666774;
        m_invA0[2][2]=+5.5235213928655478;

        m_b0tilde[0]=-3.6543931455965066;
        m_b0tilde[1]=-2.5720524267411515;
        m_b0tilde[2]=+1.8411737976218492;

        m_beta[0]=+0.0000000000000000;
        m_beta[1]=-3.7617657216935383;
        m_beta[2]=+3.7617657216935383;

        m_eta[0]=+4.2496671036869920;
        m_eta[1]=+4.1283425562285032;
        m_eta[2]=+4.1283425562285032;
    }    
    
}

/* Given the precomputed vector m_b0_tilde, and the inverse of A0, compute and 
    store coefficients of the polynomials X_j */
void IRK::SetXCoeffs() {
    
    // There are m_s lots of m_s coefficients
    m_XCoeffs.resize(m_s, std::vector<double>(m_s, 0.0));
    
    if (m_s == 1) {
        /* s=1: Coefficients for polynomial X_{1}(z) */
        m_XCoeffs[0][0]=+m_b0tilde[0];
        
    } else if (m_s == 2) {
        /* s=2: Coefficients for polynomial X_{1}(z) */
        m_XCoeffs[0][0]=+m_invA0[1][1]*m_b0tilde[0]-m_invA0[1][0]*m_b0tilde[1];
        m_XCoeffs[0][1]=-m_b0tilde[0];

        /* s=2: Coefficients for polynomial X_{2}(z) */
        m_XCoeffs[1][0]=+m_invA0[0][0]*m_b0tilde[1]-m_invA0[0][1]*m_b0tilde[0];
        m_XCoeffs[1][1]=-m_b0tilde[1];

    } else if (m_s == 3) {
        /* s=3: Coefficients for polynomial X_{1}(z) */
        m_XCoeffs[0][0]=+m_b0tilde[2]*(m_invA0[1][0]*m_invA0[2][1]-m_invA0[1][1]*m_invA0[2][0])-m_b0tilde[1]*(m_invA0[1][0]*m_invA0[2][2]-m_invA0[1][2]*m_invA0[2][0])+m_b0tilde[0]*(m_invA0[1][1]*m_invA0[2][2]-m_invA0[1][2]*m_invA0[2][1]);
        m_XCoeffs[0][1]=+m_invA0[1][0]*m_b0tilde[1]+m_invA0[2][0]*m_b0tilde[2]-m_b0tilde[0]*(m_invA0[1][1]+m_invA0[2][2]);
        m_XCoeffs[0][2]=+m_b0tilde[0];

        /* s=3: Coefficients for polynomial X_{2}(z) */
        m_XCoeffs[1][0]=+m_b0tilde[1]*(m_invA0[0][0]*m_invA0[2][2]-m_invA0[0][2]*m_invA0[2][0])-m_b0tilde[2]*(m_invA0[0][0]*m_invA0[2][1]-m_invA0[0][1]*m_invA0[2][0])-m_b0tilde[0]*(m_invA0[0][1]*m_invA0[2][2]-m_invA0[0][2]*m_invA0[2][1]);
        m_XCoeffs[1][1]=+m_invA0[0][1]*m_b0tilde[0]+m_invA0[2][1]*m_b0tilde[2]-m_b0tilde[1]*(m_invA0[0][0]+m_invA0[2][2]);
        m_XCoeffs[1][2]=+m_b0tilde[1];

        /* s=3: Coefficients for polynomial X_{3}(z) */
        m_XCoeffs[2][0]=+m_b0tilde[2]*(m_invA0[0][0]*m_invA0[1][1]-m_invA0[0][1]*m_invA0[1][0])-m_b0tilde[1]*(m_invA0[0][0]*m_invA0[1][2]-m_invA0[0][2]*m_invA0[1][0])+m_b0tilde[0]*(m_invA0[0][1]*m_invA0[1][2]-m_invA0[0][2]*m_invA0[1][1]);
        m_XCoeffs[2][1]=+m_invA0[0][2]*m_b0tilde[0]+m_invA0[1][2]*m_b0tilde[1]-m_b0tilde[2]*(m_invA0[0][0]+m_invA0[1][1]);
        m_XCoeffs[2][2]=+m_b0tilde[2];
        
    } else if (m_s == 4) {
        /* s=4: Coefficients for polynomial X_{1}(z) */
        m_XCoeffs[0][0]=+m_b0tilde[2]*(m_invA0[1][0]*m_invA0[2][1]*m_invA0[3][3]-m_invA0[1][0]*m_invA0[2][3]*m_invA0[3][1]-m_invA0[1][1]*m_invA0[2][0]*m_invA0[3][3]+m_invA0[1][1]*m_invA0[2][3]*m_invA0[3][0]+m_invA0[1][3]*m_invA0[2][0]*m_invA0[3][1]-m_invA0[1][3]*m_invA0[2][1]*m_invA0[3][0])-m_b0tilde[3]*(m_invA0[1][0]*m_invA0[2][1]*m_invA0[3][2]-m_invA0[1][0]*m_invA0[2][2]*m_invA0[3][1]-
                            m_invA0[1][1]*m_invA0[2][0]*m_invA0[3][2]+m_invA0[1][1]*m_invA0[2][2]*m_invA0[3][0]+m_invA0[1][2]*m_invA0[2][0]*m_invA0[3][1]-m_invA0[1][2]*m_invA0[2][1]*m_invA0[3][0])-m_b0tilde[1]*(m_invA0[1][0]*m_invA0[2][2]*m_invA0[3][3]-m_invA0[1][0]*m_invA0[2][3]*m_invA0[3][2]-m_invA0[1][2]*m_invA0[2][0]*m_invA0[3][3]+m_invA0[1][2]*m_invA0[2][3]*m_invA0[3][0]+m_invA0[1][3]*m_invA0[2][0]*m_invA0[3][2]-m_invA0[1][3]*m_invA0[2][2]*m_invA0[3][0])+m_b0tilde[0]*(m_invA0[1][1]*m_invA0[2][2]*m_invA0[3][3]-m_invA0[1][1]*m_invA0[2][3]*m_invA0[3][2]-m_invA0[1][2]*m_invA0[2][1]*m_invA0[3][3]+m_invA0[1][2]*m_invA0[2][3]*m_invA0[3][1]+m_invA0[1][3]*m_invA0[2][1]*m_invA0[3][2]-m_invA0[1][3]*m_invA0[2][2]*m_invA0[3][1]);
        m_XCoeffs[0][1]=+m_b0tilde[1]*(m_invA0[1][0]*m_invA0[2][2]-m_invA0[1][2]*m_invA0[2][0]+m_invA0[1][0]*m_invA0[3][3]-m_invA0[1][3]*m_invA0[3][0])-m_b0tilde[0]*(m_invA0[1][1]*m_invA0[2][2]-m_invA0[1][2]*m_invA0[2][1]+m_invA0[1][1]*m_invA0[3][3]-m_invA0[1][3]*m_invA0[3][1]+m_invA0[2][2]*m_invA0[3][3]-m_invA0[2][3]*m_invA0[3][2])-m_b0tilde[2]*(m_invA0[1][0]*m_invA0[2][1]-m_invA0[1][1]*m_invA0[2][0]-m_invA0[2][0]*m_invA0[3][3]+m_invA0[2][3]*m_invA0[3][0])-
                            m_b0tilde[3]*(m_invA0[1][0]*m_invA0[3][1]-m_invA0[1][1]*m_invA0[3][0]+m_invA0[2][0]*m_invA0[3][2]-m_invA0[2][2]*m_invA0[3][0]);
        m_XCoeffs[0][2]=+m_b0tilde[0]*(m_invA0[1][1]+m_invA0[2][2]+m_invA0[3][3])-m_invA0[2][0]*m_b0tilde[2]-m_invA0[3][0]*m_b0tilde[3]-m_invA0[1][0]*m_b0tilde[1];
        m_XCoeffs[0][3]=-m_b0tilde[0];

        /* s=4: Coefficients for polynomial X_{2}(z) */
        m_XCoeffs[1][0]=+m_b0tilde[3]*(m_invA0[0][0]*m_invA0[2][1]*m_invA0[3][2]-m_invA0[0][0]*m_invA0[2][2]*m_invA0[3][1]-m_invA0[0][1]*m_invA0[2][0]*m_invA0[3][2]+m_invA0[0][1]*m_invA0[2][2]*m_invA0[3][0]+m_invA0[0][2]*m_invA0[2][0]*m_invA0[3][1]-m_invA0[0][2]*m_invA0[2][1]*m_invA0[3][0])-m_b0tilde[2]*(m_invA0[0][0]*m_invA0[2][1]*m_invA0[3][3]-m_invA0[0][0]*m_invA0[2][3]*m_invA0[3][1]-  
                            m_invA0[0][1]*m_invA0[2][0]*m_invA0[3][3]+m_invA0[0][1]*m_invA0[2][3]*m_invA0[3][0]+m_invA0[0][3]*m_invA0[2][0]*m_invA0[3][1]-m_invA0[0][3]*m_invA0[2][1]*m_invA0[3][0])+m_b0tilde[1]*(m_invA0[0][0]*m_invA0[2][2]*m_invA0[3][3]-m_invA0[0][0]*m_invA0[2][3]*m_invA0[3][2]-m_invA0[0][2]*m_invA0[2][0]*m_invA0[3][3]+m_invA0[0][2]*m_invA0[2][3]*m_invA0[3][0]+m_invA0[0][3]*m_invA0[2][0]*m_invA0[3][2]-m_invA0[0][3]*m_invA0[2][2]*m_invA0[3][0])-m_b0tilde[0]*(m_invA0[0][1]*m_invA0[2][2]*m_invA0[3][3]-m_invA0[0][1]*m_invA0[2][3]*m_invA0[3][2]-m_invA0[0][2]*m_invA0[2][1]*m_invA0[3][3]+m_invA0[0][2]*m_invA0[2][3]*m_invA0[3][1]+m_invA0[0][3]*m_invA0[2][1]*m_invA0[3][2]-m_invA0[0][3]*m_invA0[2][2]*m_invA0[3][1]);
        m_XCoeffs[1][1]=+m_b0tilde[0]*(m_invA0[0][1]*m_invA0[2][2]-m_invA0[0][2]*m_invA0[2][1]+m_invA0[0][1]*m_invA0[3][3]-m_invA0[0][3]*m_invA0[3][1])-m_b0tilde[1]*(m_invA0[0][0]*m_invA0[2][2]-m_invA0[0][2]*m_invA0[2][0]+m_invA0[0][0]*m_invA0[3][3]-m_invA0[0][3]*m_invA0[3][0]+m_invA0[2][2]*m_invA0[3][3]-m_invA0[2][3]*m_invA0[3][2])+m_b0tilde[2]*(m_invA0[0][0]*m_invA0[2][1]-m_invA0[0][1]*m_invA0[2][0]+m_invA0[2][1]*m_invA0[3][3]-       
                            m_invA0[2][3]*m_invA0[3][1])+m_b0tilde[3]*(m_invA0[0][0]*m_invA0[3][1]-m_invA0[0][1]*m_invA0[3][0]-m_invA0[2][1]*m_invA0[3][2]+m_invA0[2][2]*m_invA0[3][1]);
        m_XCoeffs[1][2]=+m_b0tilde[1]*(m_invA0[0][0]+m_invA0[2][2]+m_invA0[3][3])-m_invA0[2][1]*m_b0tilde[2]-m_invA0[3][1]*m_b0tilde[3]-m_invA0[0][1]*m_b0tilde[0];
        m_XCoeffs[1][3]=-m_b0tilde[1];

        /* s=4: Coefficients for polynomial X_{3}(z) */
        m_XCoeffs[2][0]=+m_b0tilde[2]*(m_invA0[0][0]*m_invA0[1][1]*m_invA0[3][3]-m_invA0[0][0]*m_invA0[1][3]*m_invA0[3][1]-m_invA0[0][1]*m_invA0[1][0]*m_invA0[3][3]+m_invA0[0][1]*m_invA0[1][3]*m_invA0[3][0]+m_invA0[0][3]*m_invA0[1][0]*m_invA0[3][1]-m_invA0[0][3]*m_invA0[1][1]*m_invA0[3][0])-m_b0tilde[3]*(m_invA0[0][0]*m_invA0[1][1]*m_invA0[3][2]-m_invA0[0][0]*m_invA0[1][2]*m_invA0[3][1]-
                            m_invA0[0][1]*m_invA0[1][0]*m_invA0[3][2]+m_invA0[0][1]*m_invA0[1][2]*m_invA0[3][0]+m_invA0[0][2]*m_invA0[1][0]*m_invA0[3][1]-m_invA0[0][2]*m_invA0[1][1]*m_invA0[3][0])-m_b0tilde[1]*(m_invA0[0][0]*m_invA0[1][2]*m_invA0[3][3]-m_invA0[0][0]*m_invA0[1][3]*m_invA0[3][2]-m_invA0[0][2]*m_invA0[1][0]*m_invA0[3][3]+m_invA0[0][2]*m_invA0[1][3]*m_invA0[3][0]+m_invA0[0][3]*m_invA0[1][0]*m_invA0[3][2]-m_invA0[0][3]*m_invA0[1][2]*m_invA0[3][0])+m_b0tilde[0]*(m_invA0[0][1]*m_invA0[1][2]*m_invA0[3][3]-m_invA0[0][1]*m_invA0[1][3]*m_invA0[3][2]-m_invA0[0][2]*m_invA0[1][1]*m_invA0[3][3]+m_invA0[0][2]*m_invA0[1][3]*m_invA0[3][1]+m_invA0[0][3]*m_invA0[1][1]*m_invA0[3][2]-m_invA0[0][3]*m_invA0[1][2]*m_invA0[3][1]);
        m_XCoeffs[2][1]=+m_b0tilde[1]*(m_invA0[0][0]*m_invA0[1][2]-m_invA0[0][2]*m_invA0[1][0]+m_invA0[1][2]*m_invA0[3][3]-m_invA0[1][3]*m_invA0[3][2])-m_b0tilde[0]*(m_invA0[0][1]*m_invA0[1][2]-m_invA0[0][2]*m_invA0[1][1]-m_invA0[0][2]*m_invA0[3][3]+m_invA0[0][3]*m_invA0[3][2])-m_b0tilde[2]*(m_invA0[0][0]*m_invA0[1][1]-m_invA0[0][1]*m_invA0[1][0]+m_invA0[0][0]*m_invA0[3][3]-m_invA0[0][3]*m_invA0[3][0]+m_invA0[1][1]*m_invA0[3][3]-   
                            m_invA0[1][3]*m_invA0[3][1])+m_b0tilde[3]*(m_invA0[0][0]*m_invA0[3][2]-m_invA0[0][2]*m_invA0[3][0]+m_invA0[1][1]*m_invA0[3][2]-m_invA0[1][2]*m_invA0[3][1]);
        m_XCoeffs[2][2]=+m_b0tilde[2]*(m_invA0[0][0]+m_invA0[1][1]+m_invA0[3][3])-m_invA0[1][2]*m_b0tilde[1]-m_invA0[3][2]*m_b0tilde[3]-m_invA0[0][2]*m_b0tilde[0];
        m_XCoeffs[2][3]=-m_b0tilde[2];

        /* s=4: Coefficients for polynomial X_{4}(z) */
        m_XCoeffs[3][0]=+m_b0tilde[3]*(m_invA0[0][0]*m_invA0[1][1]*m_invA0[2][2]-m_invA0[0][0]*m_invA0[1][2]*m_invA0[2][1]-m_invA0[0][1]*m_invA0[1][0]*m_invA0[2][2]+m_invA0[0][1]*m_invA0[1][2]*m_invA0[2][0]+m_invA0[0][2]*m_invA0[1][0]*m_invA0[2][1]-m_invA0[0][2]*m_invA0[1][1]*m_invA0[2][0])-m_b0tilde[2]*(m_invA0[0][0]*m_invA0[1][1]*m_invA0[2][3]-m_invA0[0][0]*m_invA0[1][3]*m_invA0[2][1]-
                            m_invA0[0][1]*m_invA0[1][0]*m_invA0[2][3]+m_invA0[0][1]*m_invA0[1][3]*m_invA0[2][0]+m_invA0[0][3]*m_invA0[1][0]*m_invA0[2][1]-m_invA0[0][3]*m_invA0[1][1]*m_invA0[2][0])+m_b0tilde[1]*(m_invA0[0][0]*m_invA0[1][2]*m_invA0[2][3]-m_invA0[0][0]*m_invA0[1][3]*m_invA0[2][2]-m_invA0[0][2]*m_invA0[1][0]*m_invA0[2][3]+m_invA0[0][2]*m_invA0[1][3]*m_invA0[2][0]+m_invA0[0][3]*m_invA0[1][0]*m_invA0[2][2]-m_invA0[0][3]*m_invA0[1][2]*m_invA0[2][0])-m_b0tilde[0]*(m_invA0[0][1]*m_invA0[1][2]*m_invA0[2][3]-m_invA0[0][1]*m_invA0[1][3]*m_invA0[2][2]-m_invA0[0][2]*m_invA0[1][1]*m_invA0[2][3]+m_invA0[0][2]*m_invA0[1][3]*m_invA0[2][1]+m_invA0[0][3]*m_invA0[1][1]*m_invA0[2][2]-m_invA0[0][3]*m_invA0[1][2]*m_invA0[2][1]);
        m_XCoeffs[3][1]=+m_b0tilde[1]*(m_invA0[0][0]*m_invA0[1][3]-m_invA0[0][3]*m_invA0[1][0]-m_invA0[1][2]*m_invA0[2][3]+m_invA0[1][3]*m_invA0[2][2])-m_b0tilde[0]*(m_invA0[0][1]*m_invA0[1][3]-m_invA0[0][3]*m_invA0[1][1]+m_invA0[0][2]*m_invA0[2][3]-m_invA0[0][3]*m_invA0[2][2])-m_b0tilde[3]*(m_invA0[0][0]*m_invA0[1][1]-m_invA0[0][1]*m_invA0[1][0]+m_invA0[0][0]*m_invA0[2][2]-m_invA0[0][2]*m_invA0[2][0]+m_invA0[1][1]*m_invA0[2][2]-   
                            m_invA0[1][2]*m_invA0[2][1])+m_b0tilde[2]*(m_invA0[0][0]*m_invA0[2][3]-m_invA0[0][3]*m_invA0[2][0]+m_invA0[1][1]*m_invA0[2][3]-m_invA0[1][3]*m_invA0[2][1]);
        m_XCoeffs[3][2]=+m_b0tilde[3]*(m_invA0[0][0]+m_invA0[1][1]+m_invA0[2][2])-m_invA0[1][3]*m_b0tilde[1]-m_invA0[2][3]*m_b0tilde[2]-m_invA0[0][3]*m_b0tilde[0];
        m_XCoeffs[3][3]=-m_b0tilde[3];
    
    } else {
        std::cout << "WARNING: Coefficients of polynomials {X_j} not implemented for s = " << m_s << '\n';
        MPI_Finalize();
        exit(1);
    }
    
}




/* Horner's scheme for computing the action of a matrix polynomial on a vector */
void IRK::PolyAction() {
    // TODO. Ensure I check if coefficient is zero before adding a vector and/or multiplying with matrix
}