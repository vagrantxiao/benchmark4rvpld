#include "typedefs.h"
#include "../operators/dotProduct_5.h"
#include "stdio.h"

int main(){

	int i;
	int err=0;


	TRAINING_INST: for( int training_id = 0; training_id < NUM_TRAINING*5; training_id ++ ){
		err = dotProduct_5();
		err = dotProduct_5();
	}

	printf("We got %d errors\n", err);

}

