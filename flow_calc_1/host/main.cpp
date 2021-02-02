#include "typedefs.h"
#include "../operators/flow_calc_1.h"
#include "stdio.h"


int main(){

	int err=0;

	err = flow_calc_1();

	printf("We got %d errors\n", err);

}
