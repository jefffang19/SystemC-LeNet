#include "systemc.h"
#include "run_mode.h"
#include<iostream>
#include<fstream>



using namespace std;

SC_MODULE(lenet)
{	
	sc_in_clk clock;
	sc_in<bool> reset;
	sc_out<bool> rom_rd;
	sc_out<bool> ram_wr;
	sc_out<sc_uint<16> > rom_addr;
	sc_out<sc_uint<16> > ram_addr;
	sc_in<TYPE > rom_data_in;
	sc_in<TYPE > ram_data_in;
	sc_out<TYPE > ram_data_out;
	sc_out<TYPE > result;
	sc_out<bool> valid;
	
	
	ifstream fin;
	
	//strategy
	//input -> conv1

	//store img to ram (done)

	//read kernal 1 from rom and image from ram both 1 element at a time, multiply(or add in bias case)
	//in next cycle (done)

	//send reuslt to ram (done)

	//slide window (done)

	//result might be wrong!!

	//switch 2 next filter

	void run_lenet();

	void img_write_to_ram();
	void img_read_from_ram();
	void data_store_to_ram(TYPE value);
	void read_kernal();
	
	
	SC_CTOR(lenet)
	{
		fin.open(INPUT_FILE);
		SC_METHOD(run_lenet);
		sensitive << clock.neg();
	}
};

