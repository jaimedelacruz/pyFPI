#ifndef INVERTHPP
#define INVERTHPP
/* ---
   C++ wrappers to perform the inversion in parallel using OpenMP 

   Coded by J. de la Cruz Rodriguez (ISP-SU, 2025)
   --- */

#include "myTypes.hpp"
#include "fpi.hpp"

namespace fpi{

  void invert_hre_crisp(long const ny, long const nx, long const npar, long const nwav, long const nfts, \
			const ft* const fts_x, const ft* const fts_y, const float* const d, ft* const par, \
			float* const syn, const ft* const wav,		\
			std::vector<fpi::FPI*> &fpis, const ft* const sig, const ft* const tw, const int* const fixed,
			int const fpi_method, bool const no_pref, const float* const ecl, const float* const erl);


  void invert_lre_crisp(long const ny, long const nx, long const npar, long const nwav, \
			const float* const d, ft* const par, float* const syn, const ft* const wav,\
			std::vector<fpi::FPI*> &fpis, const ft* const sig, const ft* const tw, \
			const float* const pref, const ft* const ech, const ft* const erh,
			int const fpi_method);



  void invert_lre_crisp_laser(long const ny, long const nx, long const npar, long const nwav, \
			      const float* const d, ft* const par, float* const syn, const ft* const wav, \
			      std::vector<fpi::FPI*> &fpis, const ft* const sig, const ft* const tw );


  void invert_hre_crisp_laser(long const ny, long const nx, long const npar, long const nwav, \
			      const float* const d, ft* const par, float* const syn, const ft* const wav, \
			      std::vector<fpi::FPI*> &fpis, const ft* const sig, const ft* const tw );
  
  

  void invert_all_crisp(long const ny, long const nx, long const npar, long const nwavh, long const nwavl, long const nfts, \
			const ft* const fts_x, const ft* const fts_y, const ft* const fts_yl, const float* const dh, const float* const dl,
			ft* const par, float* const syn, const ft* const wavh, const ft* const wavl,	\
			std::vector<fpi::FPI*> &fpis, const ft* const sigh, const ft* const sigl, const ft* const tw, const int* const fixed,
			int const fpi_method, ft const dwgrid);

  
}


#endif

