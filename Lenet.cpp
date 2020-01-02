#include "Lenet.h"
#include <vector>
#include <cmath>
#define KERNAL_SIZE 25
#define IMG_SIZE 784

static bool image_read = false;
static TYPE img;
static int ram_tracker = 0;
static int rom_tracker = 0;
static int conv_tracker = 0;
static TYPE rom_data_in_tmp;
static bool request_to_rom = false;
static TYPE kernal[KERNAL_SIZE]; 
static TYPE bias; 

void img_read(){
    if(!image_read){
        fin.open("Pattern/input1.txt");
        image_read = true;
    }
    fin >> img;
    ++ram_tracker;
    ram_wr = true;
    ram_data_in = img;
    if(ram_tracker==IMG_SIZE) f.close();
}
//use this function to read kernal value and store into kernal[KERNAL_SIZE] and bias[]
//Caution require setting rom_tracker manually
void read_kernal(){
    if(request_to_rom) kernal[rom_tracker++] = rom_data_in_tmp;
    else request_to_rom = true;
}

void lenet::run_lenet(){
    if(ram_tracker<IMG_SIZE) img_read();

    if(rom_tracker < KERNAL_SIZE) read_kernal();
    else if(rom_tracker == 25){
        rom_data_in_tmp = rom_data_in;
        bias = rom_data_in_tmp;
    }
    //start doing convolution
    else if(rom_tracker == 26 && conv_tracker < 24*24){
        int k = 0; //iterator for img
       int sum = 0;
       int img_len = sqrt(IMG_SIZE);
       int kernal_len = sqrt(KERNAL_SIZE);
       for(int i=0;i<KERNAL_SIZE;++i){
           sum += img[k++] * kernal[i];
           if(k%kernal_len == 0) k = k - kernal_len + img_len;  
       }
       sum += bias;
       ++conv_tracker; 
    }
}
