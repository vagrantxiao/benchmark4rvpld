#include "typedefs.h"
#include "../operators/dotProduct_5.h"
//#include "input.h"
//#include "output.h"
#include <iostream>
#include <queue>

//void data_gen(hls::stream<databus_t> &Output_1)
//{
//	for(int i=0; i<446464; i++){
//		Output_1.write(input[i]);
//	}
//}



int main(){

	int i;
	int err=0;

	//data_gen(Input_1);

	TRAINING_INST: for( int training_id = 0; training_id < NUM_TRAINING*5; training_id ++ ){
		err = dotProduct_5();
		err = dotProduct_5();
	}

	//for(i=0; i<446464; i++){
	//	databus_t out_tmp = Output_1.read();
	//	if(out_tmp != output[i]){ err++; }
	//}

	printf("We got %d errors\n", err);

}


/*
// CPP program to illustrate
// Implementation of push() function
#include <iostream>
#include <queue>
using namespace std;

int main()
{
    // Empty Queue
    queue<int> myqueue;
    myqueue.push(0);
    myqueue.push(1);
    myqueue.push(2);

    // Printing content of queue
    while (!myqueue.empty()) {
        cout << ' ' << myqueue.front();
        myqueue.pop();
    }
}

*/
