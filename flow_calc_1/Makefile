
host_dir=./host
operators_dir=./operators
operators_src=$(wildcard $(operators_dir)/*.cpp)
operators=$(basename $(notdir $(operators_src)))
operators_obj=$(addsuffix .o, $(basename $(notdir $(operators_src))))
C_FLAG=-O3

run: Test_host
	./Test_host

Test_host: $(operators_obj) $(host_dir)/*
	g++ -o $@ $(host_dir)/*.cpp $(operators_obj) $(C_FLAG)

$(operators_obj):%.o:$(operators_dir)/%.cpp $(host_dir)/*
	g++ -o $@ -c $<  $(C_FLAG)


clean:
	rm -rf ./Test_host ./*.o
