/*===============================================================*/
/*                                                               */
/*                          kernel.h                             */
/*                                                               */
/*        Defines types and constants for host function          */
/*                                                               */
/*===============================================================*/

#ifndef __TYPEDEFS_H__
#define __TYPEDEFS_H__
//#include "ap_fixed.h"
const int MAX_HEIGHT = 436;
const int MAX_WIDTH = 1024;
#include "ap_int.h"
#include "ap_fixed.h"
typedef ap_uint<32> databus_t;
typedef ap_uint<128> bit128;
typedef ap_uint<160> bit160;

typedef ap_uint<288> widebus_t;
// define these constants so they can be used in pragma
const int max_width = MAX_WIDTH;
const int default_depth = MAX_WIDTH;



typedef ap_fixed<17,9> input_t;
typedef ap_fixed<32,13> pixel_t;
typedef ap_fixed<48,27> outer_pixel_t;
typedef ap_fixed<96,56> calc_pixel_t;
typedef ap_fixed<192,112> calc_pixel_t_x;
typedef ap_fixed<32,13> vel_pixel_t;
	
typedef struct{
	pixel_t x;
	pixel_t y;
	pixel_t z;
}gradient_t;

typedef struct{
    outer_pixel_t val[6];
}outer_t; 

typedef struct{
    outer_pixel_t val[3];
}outer_half_t;

typedef struct{
    outer_pixel_t val[6];
}tensor_t;

typedef struct{
    outer_pixel_t val[3];
}tensor_half_t;

typedef struct{
    vel_pixel_t x;
    vel_pixel_t y;
}velocity_t;

  #include "ap_int.h"
  // for data packing
  typedef ap_uint<64> frames_t;
  typedef ap_uint<32> stdio_t;


  // dataset information
  const int NUM_FEATURES  = 1024;
  const int NUM_SAMPLES   = 5000;
  const int NUM_TRAINING  = 4500;
  const int NUM_TESTING   = 500;
  const int STEP_SIZE     = 60000;
  const int NUM_EPOCHS    = 5;
  const int DATA_SET_SIZE = NUM_FEATURES * NUM_SAMPLES;



    // embedded platforms have less off-chip bandwidth
    #define VFTYPE_WIDTH  64
    #define VDTYPE_WIDTH  64

  #define PAR_FACTOR 32
  #define PAR_FACTOR_DEC 4
  // datatypes for accelerator

    // features / parameters
    typedef ap_uint<128> bit128;
    typedef ap_uint<32> bit32;
    #define FTYPE_TWIDTH 32
    #define FTYPE_IWIDTH 13
    typedef ap_fixed<FTYPE_TWIDTH,FTYPE_IWIDTH> FeatureType;
    typedef ap_uint<VFTYPE_WIDTH>               VectorFeatureType;
    const int F_VECTOR_SIZE = VFTYPE_WIDTH / FTYPE_TWIDTH;
    // training data
    #define DTYPE_TWIDTH 16
    #define DTYPE_IWIDTH 4
    typedef ap_fixed<DTYPE_TWIDTH,DTYPE_IWIDTH>  DataType;
    typedef ap_uint<VDTYPE_WIDTH>                VectorDataType;
    const int D_VECTOR_SIZE = VDTYPE_WIDTH / DTYPE_TWIDTH;
    // label
    #define LTYPE_WIDTH   8
    #define VLTYPE_WIDTH  32
    typedef ap_uint<LTYPE_WIDTH>                 LabelType;
    typedef ap_uint<VLTYPE_WIDTH>                VectorLabelType;
    const int L_VECTOR_SIZE = VLTYPE_WIDTH / LTYPE_WIDTH;

    // datatypes for the LUT
    #define LUTOUT_TWIDTH     12
    #define LUTOUT_IWIDTH     2
    #define LUTIN_TWIDTH      12
    #define LUTIN_IWIDTH      4
    //typedef ap_ufixed<32, 20> TmpFixed;
    typedef ap_uint<LUTIN_TWIDTH> IdxFixed;
    typedef ap_fixed<LUTIN_TWIDTH, LUTIN_IWIDTH> LutInFixed;
    typedef ap_fixed<LUTOUT_TWIDTH, LUTOUT_IWIDTH> LutOutFixed;


#endif
