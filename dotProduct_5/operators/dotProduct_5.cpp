#include "../host/typedefs.h"
#include "../host/input1.h"
#include "../host/input2.h"
#include "../host/output1.h"
#include "../host/output2.h"


int dotProduct_5()
{

#pragma HLS INTERFACE ap_hs port=Input_1
#pragma HLS INTERFACE ap_hs port=Input_2
#pragma HLS INTERFACE ap_hs port=Output_1
#pragma HLS INTERFACE ap_hs port=Output_2
  const int unroll_factor = PAR_FACTOR_DEC;
  static FeatureType param[NUM_FEATURES/8];
  FeatureType grad[NUM_FEATURES/8];
  static DataType feature[NUM_FEATURES/8];
  FeatureType scale;
  FeatureType prob;
  static int err_cnt = 0;

  #pragma HLS array_partition variable=param cyclic factor=unroll_factor
  #pragma HLS array_partition variable=feature cyclic factor=unroll_factor
  #pragma HLS array_partition variable=grad cyclic factor=unroll_factor
  static int odd_even = 0;
  static int num_train = 0;
  static int epoch;
  static int sb = 0;
  static LabelType training_label;
  static int in1_cnt=0;
  static int in2_cnt=0;
  static int out1_cnt=0;
  static int out2_cnt=0;

#ifdef RISCV
	  print_str("sb=");
	  print_dec(sb);
	  sb++;
	  print_str("\n");
#endif

  if(odd_even == 0){
	  //training_label(7,0) = Input_1.read();
	  training_label(7,0) = input1[in1_cnt];
	  in1_cnt++;
	  //printf("0x%08x,\n", training_label.to_int());

	  READ_TRAINING_DATA: for (int i = 0; i < NUM_FEATURES / D_VECTOR_SIZE / 8; i ++ )
	  //                                      1024           4
	  {
#pragma HLS PIPELINE II=1
		bit32 tmp_data;
		//tmp_data = Input_1.read();
		tmp_data = input1[in1_cnt];
		in1_cnt++;
		//printf("0x%08x,\n", tmp_data.to_int());

		feature[i * D_VECTOR_SIZE + 0](DTYPE_TWIDTH-1, 0) = tmp_data.range(15,  0);
		feature[i * D_VECTOR_SIZE + 1](DTYPE_TWIDTH-1, 0) = tmp_data.range(31, 16);
		//tmp_data = Input_1.read();
		tmp_data = input1[in1_cnt];
		in1_cnt++;
		//printf("0x%08x,\n", tmp_data.to_int());
		feature[i * D_VECTOR_SIZE + 2](DTYPE_TWIDTH-1, 0) = tmp_data.range(15,  0);
		feature[i * D_VECTOR_SIZE + 3](DTYPE_TWIDTH-1, 0) = tmp_data.range(31, 16);
	  }


	  FeatureType result = 0;
	  DOT: for (int i = 0; i < NUM_FEATURES / PAR_FACTOR_DEC / 8; i++)
	  {
		#pragma HLS PIPELINE II=1
		DOT_INNER: for(int j = 0; j < PAR_FACTOR_DEC; j++)
		{
		  FeatureType term = param[i*PAR_FACTOR_DEC+j] * feature[i*PAR_FACTOR_DEC+j];
		  result = result + term;
		}
	  }
	  //Output_1.write(result.range(31,0));
	  if((unsigned int) result.range(31,0) != output1[out1_cnt]){
		  err_cnt++;
	  }
	  out1_cnt++;

	  //printf("0x%08x,\n", (unsigned int) result(31,0));
	  odd_even = 1;
	  return err_cnt;
  }else{
	  //prob(31,0) = Input_2.read();
	  prob(31,0) = input2[in2_cnt];
	  in2_cnt++;
	  //printf("0x%08x,\n", (unsigned int) prob(31,0));
	  scale = prob - training_label;

	  GRAD: for (int i = 0; i < NUM_FEATURES / PAR_FACTOR_DEC / 8; i++)
	  {
		#pragma HLS PIPELINE II=1
		GRAD_INNER: for (int j = 0; j < PAR_FACTOR_DEC; j++)
		  grad[i*PAR_FACTOR_DEC+j] = (scale * feature[i*PAR_FACTOR_DEC+j]);
	  }

	  FeatureType step = STEP_SIZE;
	  UPDATE: for (int i = 0; i < NUM_FEATURES / PAR_FACTOR_DEC/8; i++)
	  {
		#pragma HLS PIPELINE II=1
		UPDATE_INNER: for (int j = 0; j < PAR_FACTOR_DEC; j++){
			FeatureType tmp;
			tmp = (-step) * grad[i*PAR_FACTOR_DEC+j];
			param[i*PAR_FACTOR_DEC+j] = param[i*PAR_FACTOR_DEC+j] + tmp;
		}
	  }

	  num_train++;
	  if(num_train==NUM_TRAINING){
		  num_train = 0;
		  epoch++;
	  }
	  if(epoch==5){
		  STREAM_OUT: for (int i = 0; i < NUM_FEATURES / F_VECTOR_SIZE / 8; i ++ )
		  {
			#pragma HLS pipeline II=1
			//VectorFeatureType tmp_theta = 0;
			bit32 tmp_data1;
			bit32 tmp_data2;

			tmp_data1(31,0) = param[i * F_VECTOR_SIZE + 0].range(FTYPE_TWIDTH-1, 0);
			tmp_data2(31,0)  = param[i * F_VECTOR_SIZE + 1].range(FTYPE_TWIDTH-1, 0);

			//Output_2.write(tmp_theta.range(31,0));
			if((unsigned int) tmp_data1.range(31,0) != output2[out2_cnt]){
				err_cnt++;
			}
			out2_cnt++;
			//printf("0x%08x,\n", (unsigned int) tmp_theta.range(31,0));
			//Output_2.write(tmp_theta.range(63,32));
			if((unsigned int) tmp_data2.range(31,0) != output2[out2_cnt]){
			  err_cnt++;
			}
			out2_cnt++;
			//printf("0x%08x,\n", (unsigned int) tmp_theta.range(63,32));
		  }
	  }
	  odd_even = 0;
	  return err_cnt;
  }
}
