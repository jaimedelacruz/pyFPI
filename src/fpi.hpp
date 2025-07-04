#ifndef FPIHPP
#define FPIHPP

/* ---

   FPI class to generate a dual etalon transmission profile
   and its derivatives relative to the cavity and reflectivity
   errors. There are three types of routines, based on name termination:
   
   -  ray: evaluates the profile for a single ray at a given angle.
   - conv: fast approximation that accounts for the slightly converging beam
           at F#165 (for CRISP). It assumes a symmetric beam. Based on
	   Scharmer's ANA routines, with optimizations for speed.
   - full: more accurate calculation, including the tilt of the LRE.
           The angle selection is based on sampling the pupil and performing
	   a histogram of the angular values. This calculation is only
	   done once (and stored for subsequent calculations). Based on Scharmer's
	   ANA routines.
           
   The "individual"-named function return the profiles of the LRE and HRE in
   separate output arrays, instead of returning the product of the two. These
   are used in the LRE-scan dat fitting.

   The class methods are implemented in fpi.cpp and fpi_individual.cpp.
	   
   Coded by J. de la Cruz Rodriguez (ISP-SU, 2025)

   
   References:
       Scharmer (2006);
       de la Cruz Rodriguez (2010) (self-study numerical PhD "course" at SU);
       Scharmer, de la Cruz Rodriguez et al. (2013);
       

   Comments:
       The derivatives can be trivially obtained by deriving each equation
       and propagating them with the chain rule. They are nearly identical to
       the finite difference ones, but hopefully faster to compute.

       
   --- */

#include <memory>

#include "myTypes.hpp"
#include "CRISP_ref.hpp"
#include "math.hpp"
#include "fpi_helper.hpp"

namespace fpi{
  
  // ********************************************************************* //

  inline constexpr const ft PI = 3.1415926535897932384626433832;
  inline constexpr const ft two_pi = 2.0 * PI;
  
  // ******************************************************************** //

  //constexpr inline const int NRAYS_HR = 11;
  //constexpr inline const int NRAYS_LR = 11;
  constexpr inline const int NRAYS = 7;
  constexpr inline const int NL = 2*256;
  constexpr inline const int NR = 248;
  
  // ******************************************************************** //
  
  class FPI{
  public:
    ft cw;
    ft FR;

    ft hc;
    ft lc;
    ft lc_tilted;
    
    ft hfsr;
    ft lfsr;

    ft lr;
    ft hr;

    ft BlueShift;
    int const NRAYS_HR;
    int const NRAYS_LR;
    
    std::array<ft,NRAYS> calp;
    std::array<ft,NRAYS> wng;

    Arr2D<ft> n_betah;
    Arr1D<ft> betah_lr;
    Arr1D<ft> betah_hr;
    
    std::unique_ptr<mth::fftconv1D<ft>> convolver;
    std::unique_ptr<mth::fftconv1D<ft>> convolver2;
    
    
    // -------------------------------------------------------------- //
    
    FPI(ft const icw, ft const iFR, ft const ishr, ft const islr, int const iNRAYS_HR, int const iNRAYS_LR);

    // -------------------------------------------------------------- //

    ~FPI();
    
    // -------------------------------------------------------------- //

    void dual_fpi_conv(int const N1, const ft* const tw,
		       ft* const tr,
		       ft const erh, ft const erl,
		       ft const ech, ft const ecl, bool const normalize)const;
    
    // -------------------------------------------------------------- //

    void dual_fpi_conv_individual(int const N1, const ft* const tw,
				  ft* const htr, ft* const ltr,
				  ft const erh, ft const erl,
				  ft const ech, ft const ecl,
				  bool const normalize_ltr,
				  bool const normalize_ht)const;
    
    // -------------------------------------------------------------- //

    void dual_fpi_conv_individual_der(int const N1, const ft* const tw,
				      ft* const htr, ft* const ltr,
				      ft* const dtr,
				      ft const erh, ft const erl,
				      ft const ech, ft const ecl,
				      bool const normalize_ltr,
				      bool const normalize_ht)const;
    
    // -------------------------------------------------------------- //

    void dual_fpi_conv_der(int const N1, const ft* const tw,
			   ft* const tr, ft* const dtr,
			   ft const erh, ft const erl,
			   ft const ech, ft const ecl, bool const normalize)const;
    
    // -------------------------------------------------------------- //


    void dual_fpi_ray(int const N1, const ft* const tw,
		      ft* const tr,
		      ft const erh, ft const erl,
		      ft const ech, ft const ecl,
		      ft const angle, bool const normalize)const;
        
    // -------------------------------------------------------------- //

    void dual_fpi_ray_der(int const N1, const ft* const tw,
			  ft* const tr, ft* const dtr,
			  ft const erh, ft const erl,
			  ft const ech, ft const ecl,
			  ft const angle, bool const normalize)const;
    
    // -------------------------------------------------------------- //

    ft getBlueShift()const;

    // -------------------------------------------------------------- //

    ft getFWHM()const;
    
    // -------------------------------------------------------------- //

    ft const getFSR()const;
    
    // -------------------------------------------------------------- //

    void init_convolver(int const ndata, int const npsf);
    
    // -------------------------------------------------------------- //

    void init_convolver2(int const ndata, int const npsf);
    
    // -------------------------------------------------------------- //
    
    void dual_fpi_full(int const N1, const ft* const tw, ft* const tr,
		       ft const erh, ft const erl, ft const ech,
		       ft const ecl, bool const normalize)const;

    // -------------------------------------------------------------- //
    
    void dual_fpi_full_der(int const N1, const ft* const tw, ft* const tr,
			   ft* const dtr, ft const erh, ft const erl, ft const ech,
			   ft const ecl, bool const normalize)const;
    
    // -------------------------------------------------------------- //

    void dual_fpi_full_individual(int const N1, const ft* const tw, ft* const htr, ft* const lht,
				  ft const erh, ft const erl, ft const ech,
				  ft const ecl,
				  bool const normalize_ltr,
				  bool const normalize_htr)const;

    // -------------------------------------------------------------- //
    
    void dual_fpi_full_individual_der(int const N1, const ft* const tw, ft* const htr, ft* const lht,
				      ft* const dtr, ft const erh, ft const erl, ft const ech,
				      ft const ecl,
				      bool const normalize_ltr,
				      bool const normalize_htr)const;
    
    // -------------------------------------------------------------- //

    void dual_fpi_ray_individual(int const N1, const ft* const tw,
				 ft* const htr, ft* const ltr,
				 ft const erh, ft const erl,
				 ft const ech, ft const ecl,
				 ft const angle, bool const normalize_ltr,
				 bool const normalize_htr)const;

    // -------------------------------------------------------------- //

    void dual_fpi_ray_individual_der(int const N1, const ft* const tw,
				     ft* const htr, ft* const ltr, ft* const dtr,
				     ft const erh, ft const erl,
				     ft const ech, ft const ecl,
				     ft const angle, bool const normalize_ltr,
				     bool const normalize_htr)const;
    
    // -------------------------------------------------------------- //

    void set_reflectivities(ft const ihr, ft const ilr);

    // -------------------------------------------------------------- //

    ft get_HRE_reflectivity()const;

    // -------------------------------------------------------------- //

    ft get_LRE_reflectivity()const;

    // -------------------------------------------------------------- //

    
  };
  

}

#endif
