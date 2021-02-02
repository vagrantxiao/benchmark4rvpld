#include "../host/typedefs.h"
#include "../host/input1.h"
#include "../host/input2.h"
#include "../host/output.h"



int flow_calc_1()
{
#pragma HLS interface ap_hs port=Input_1
#pragma HLS interface ap_hs port=Output_1
#pragma HLS interface ap_hs port=Input_2
  static float buf;
  static int in1_cnt=0;
  static int in2_cnt = 0;
  static int out_cnt = 0;
  static int err_cnt = 0;
  FLOW_OUTER: for(int r=0; r<MAX_HEIGHT; r++)
  {
#ifdef RISCV
	  print_str("r=");
	  print_dec(r);
	  print_str("\n");
#endif
    FLOW_INNER: for(int c=0; c<MAX_WIDTH; c++)
    {
#ifdef RISCV1
	  print_str("r=");
	  print_dec(r);
	  print_str(", c=");
	  print_dec(c);
	  print_str("\n");
#endif
      #pragma HLS pipeline II=1
      tensor_t tmp_tensor;
      stdio_t widetemp1, widetemp2, widetemp3, widetemp4, widetemp5;
      //widetemp1(31, 0) = Input_1.read();
      //widetemp2(31, 0) = Input_1.read();
      //widetemp3(31, 0) = Input_1.read();
      //widetemp4(31, 0) = Input_1.read();
      //widetemp5(31, 0) = Input_1.read();

      widetemp1(31, 0) = input1[in1_cnt++];
      widetemp2(31, 0) = input1[in1_cnt++];
      widetemp3(31, 0) = input1[in1_cnt++];
      widetemp4(31, 0) = input1[in1_cnt++];
      widetemp5(31, 0) = input1[in1_cnt++];

      //printf("0x%08x,\n", widetemp1.to_int());
      //printf("0x%08x,\n", widetemp2.to_int());
      //printf("0x%08x,\n", widetemp3.to_int());
      //printf("0x%08x,\n", widetemp4.to_int());
      //printf("0x%08x,\n", widetemp5.to_int());


      tmp_tensor.val[0](31, 0) = widetemp1(31, 0);
      tmp_tensor.val[0](47,32) = widetemp2(15, 0);
      tmp_tensor.val[1](15, 0) = widetemp2(31,16);
      tmp_tensor.val[1](47,16) = widetemp3(31, 0);
      tmp_tensor.val[2](31, 0) = widetemp4(31, 0);
      tmp_tensor.val[2](47,32) = widetemp5(15, 0);


      //widetemp1(31, 0) = Input_2.read();
      //widetemp2(31, 0) = Input_2.read();
      //widetemp3(31, 0) = Input_2.read();
      //widetemp4(31, 0) = Input_2.read();
      //widetemp5(31, 0) = Input_2.read();

      widetemp1(31, 0) = input2[in2_cnt++];
      widetemp2(31, 0) = input2[in2_cnt++];
      widetemp3(31, 0) = input2[in2_cnt++];
      widetemp4(31, 0) = input2[in2_cnt++];
      widetemp5(31, 0) = input2[in2_cnt++];

      //printf("0x%08x,\n", widetemp1.to_int());
      //printf("0x%08x,\n", widetemp2.to_int());
      //printf("0x%08x,\n", widetemp3.to_int());
      //printf("0x%08x,\n", widetemp4.to_int());
      //printf("0x%08x,\n", widetemp5.to_int());


      tmp_tensor.val[3](31, 0) = widetemp1(31, 0);
      tmp_tensor.val[3](47,32) = widetemp2(15, 0);
      tmp_tensor.val[4](15, 0) = widetemp2(31,16);
      tmp_tensor.val[4](47,16) = widetemp3(31, 0);
      tmp_tensor.val[5](31, 0) = widetemp4(31, 0);
      tmp_tensor.val[5](47,32) = widetemp5(15, 0);


      if(r>=2 && r<MAX_HEIGHT-2 && c>=2 && c<MAX_WIDTH-2)
      {
	      calc_pixel_t t1 = (calc_pixel_t) tmp_tensor.val[0];
	      calc_pixel_t t2 = (calc_pixel_t) tmp_tensor.val[1];
	      calc_pixel_t t4 = (calc_pixel_t) tmp_tensor.val[2];
	      calc_pixel_t t5 = (calc_pixel_t) tmp_tensor.val[4];
	      calc_pixel_t t6 = (calc_pixel_t) tmp_tensor.val[5];
	      calc_pixel_t denom = t1*t2-t4*t4;
	      calc_pixel_t numer0 = t6*t4-t5*t2;

	      if(denom != 0)
              {
	          buf =(float) numer0 / (float) denom;
        	  //buf = (ap_fixed<64,56>) numer0 / (ap_fixed<64,56>) denom;
	      }
	      else
	      {
		      buf = 0;
	      }
      }
      else
      {
        buf = 0;
      }
      stdio_t tmpframe;
      vel_pixel_t tmpvel;
      tmpvel = (vel_pixel_t)buf;
      tmpframe(31,0) = tmpvel(31,0);
      //Output_1.write(tmpframe);
      if(tmpframe != output[out_cnt++]){ err_cnt++; }
    }
  }
  return err_cnt;
}
