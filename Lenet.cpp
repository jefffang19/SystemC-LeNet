#include "Lenet.h"
#include <vector>
#include <cmath>
#define KERNAL_SIZE 25
#define IMG_SIZE 784
#define CONV_LAYER1 6
#define CONV_LAYER2 16
#define FC_LAYER1 128


//ram stuff
static bool image_write = true;
static bool image_read = false;
static TYPE img;
static sc_uint<16> ram_tracker = 0;
static bool ram_debuging = false;
static sc_uint<16> result_to_ram; //pointer of storing result to ram
static sc_uint<16> begin_of_conv_in_ram; //pointer of the start of convolution result in ram
static sc_uint<16> begin_of_pool_in_ram; //pointer of the start of pooling result in ram
static bool dedicate_to_storing = false;

//rom stuff
static sc_uint<16> rom_tracker = 0;
static bool request_to_rom = false;
static TYPE kernal;
static sc_uint<16> filter_start = 0; //decide which filter to use

//operation counter
//in conv : count multiply operation equals to filter size
//in pooling : count 4 compare
static int op_cnt; 
static int total_op_counter = 0;

//use for convolution
static int filter_cnt1 = 1; //count first conv layer
static int filter_cnt2 = 0; //count second conv layer
static int depth_cnt = 0;
static bool start_conv = false;
static bool start_conv2 = false;
static int start_of_conv = 0; //starting position of convolution
static TYPE sum = 0;
static int kernal_len = sqrt(KERNAL_SIZE);
static int img_len = sqrt(IMG_SIZE);
static int count_slide = 0;

//pool
static bool start_pooling = false;
static bool start_pooling2 = false;
static TYPE max_pool_value;

//fc layers
static bool signal_fc = false;
static bool start_fclayer1 = false;
static bool start_fclayer2 = false;
static bool start_fclayer3 = false;


//debug
static int counting = 0;
static int ff = 0;
static sc_uint<16> start;
static sc_uint<16> end;
static int cut;

TYPE relu(TYPE input);

void debug_register(){
    cout <<"ram:\n" << "image write: " << image_write << endl;
    cout << "image read: " << image_read << endl;
    cout << "img: " << img << endl;
    cout << "ram tracker: " << ram_tracker << endl;
    cout << "ram debugging: " << ram_debuging << endl;
    cout << "result_to_ram: " << result_to_ram << endl;
    cout << "begin_of_conv_in_ram: " <<  begin_of_conv_in_ram << endl;
    cout << "begin_of_pool_in_ram: " << begin_of_pool_in_ram << endl;
    cout << "dedicate_to_storing: " << dedicate_to_storing << endl;
    cout << "\nrom:\n" << "rom_tracker: " << rom_tracker << endl;
    cout << "request_to_rom: " << request_to_rom << endl;
    cout << "kernal: " << kernal << endl;
    cout << "filter_start: " << filter_start << endl;
    cout << "\noperation counter\n" << "op_cnt: " << op_cnt << endl;
    cout << "total_op_counter: " << total_op_counter << endl;
    cout << "\nuse for convolution:\n" << "filter_cnt1: " << filter_cnt1 << endl;
    cout << "start_conv: " << start_conv << endl;
    cout << "start_of_conv: " << start_of_conv << endl;
    cout << "sum: " << sum << endl;
    cout << "kernal_len: " << kernal_len << endl;
    cout << "img_len: " << img_len << endl;
    cout << "count_slide: " << count_slide << endl;
    cout << "\n\npool:\n" << start_pooling << endl;
    cout << "max_pool_value: " << max_pool_value << endl;
    cout << "--------------------------------------------------\n\n";
}

void lenet::run_lenet(){
    if(!ram_debuging || dedicate_to_storing){
        //first few cycle dedicate to readin image and store to ram
        if(ram_tracker<IMG_SIZE && image_write) img_write_to_ram();
        
        //read 1 piece of kernal and 1 piece of img. (after image fully store to ram)
        if(ram_tracker<IMG_SIZE && image_read) img_read_from_ram();
        //else if(start_conv2 && image_read) img_read_from_ram();

        if((rom_tracker - filter_start) < KERNAL_SIZE +2 && image_read && !start_pooling && !start_conv2) read_kernal();
        
        //start doing convolution
        if(start_conv && !start_conv2 && !dedicate_to_storing){
            if(op_cnt++ < KERNAL_SIZE){ //convolution operation
                sum += img * kernal;
                ram_tracker++;
                //cout << sum << " "; //debug
                //cout << "conv cnt:" << op_cnt << " " << img << "\t" << kernal << endl; //debug
                //cout << kernal << " "; //debug
            }
            else{ //add bias
                sum += kernal;
                
                //activation function
                sum = relu(sum);

                ++total_op_counter;

                //if(sum!=0) cout << "debug:" << sum << " " << result_to_ram << " " << total_op_counter << endl;

                //store to ram
                data_store_to_ram(sum);

                //cout << "Finish: " << sum << endl << endl; //debug
                //image_read = false; //debug
                //start_conv = false; //deubg
                //debug
                
                if(total_op_counter==576){
                    image_read = false; 
                    start_conv = false;

                    //ram debug setting
                    //ram_debuging = true;
                }

                //clear buffers
                op_cnt = 0;
                sum = 0;
                count_slide = 0;

                //calculate the next starting point
                ++start_of_conv;
                //check if reach image's edge - filter length
                if(start_of_conv % img_len == img_len - (kernal_len - 1)) start_of_conv+=(kernal_len - 1);

                //set the next starting point
                ram_tracker = start_of_conv;
                
                
            }

            //if reach the filter's length
            if(count_slide == kernal_len -1){
                ram_tracker = ram_tracker - kernal_len + img_len;
                //cout << "debug_c:" << ram_tracker << " "; //debug
                count_slide = 0;
            }
            else if(!dedicate_to_storing) ++count_slide;

            if(!dedicate_to_storing) ram_addr = ram_tracker;

            //if(total_op_counter<5) cout << "Rt=" << ram_tracker << " " << start_of_conv << " " << sum << endl;
        }
        else if(start_conv2 && !dedicate_to_storing){
            if(op_cnt++ < KERNAL_SIZE){ //convolution operation
                img = ram_data_in;
                kernal = rom_data_in;
                sum += img * kernal;
                //cout << sum << " "; //debug
                //cout << "conv cnt:" << op_cnt << " rom tracker: " << rom_tracker << " ram tracker: " << ram_tracker << " " << " " << img << "\t" << kernal << endl; //debug
                //cout << kernal << " "; //debug
            }
            if(depth_cnt < 6 && op_cnt == KERNAL_SIZE){
               //clear buffers
                op_cnt = 0;
                count_slide = 0;

                //calculate the next starting point
                start_of_conv += 144;

                //change the start of the filter
                filter_start += 25;
                rom_tracker = filter_start;

                //set the next starting point
                ram_tracker = start_of_conv - 1;

                //depth deeper
                ++depth_cnt; 
            }
            else if(op_cnt == KERNAL_SIZE){ //add bias
                //debug
                //cout << "debug: enter 6 stage.\n";
                sum += kernal;

                //activation function
                sum = relu(sum);

                ++total_op_counter;

                //if(sum!=0) cout << "debug:" << sum << " " << result_to_ram << " " << total_op_counter << endl;

                //store to ram
                data_store_to_ram(sum);

                //cout << "Finish: " << sum << endl << endl; //debug
                //image_read = false; //debug
                //start_conv = false; //deubg
                //debug

                if(total_op_counter<64){
                    //change the start of the filter
                    filter_start -= 125;
                    rom_tracker = filter_start;
                }
                else if(total_op_counter==64){
                    start_conv = false;

                    //change the start of the filter
                    filter_start += 26;
                    rom_tracker = filter_start;

                    //ram debug setting
                    //ram_debuging = true;
                    //img_len = 64;
                    //kernal_len = 8;
                }
                //cout << "DEubg: filterstart: " << filter_start << endl;

                //clear buffers
                op_cnt = 0;
                sum = 0;
                count_slide = 0;
                //reset kernal depth
                depth_cnt = 1;

                //calculate the next starting point
                start_of_conv = start_of_conv - 720 + 1;

                //check if reach image's edge - filter length
                if((start_of_conv-begin_of_conv_in_ram) % img_len == img_len - (kernal_len - 1)) start_of_conv+=(kernal_len - 1);

                //set the next starting point
                ram_tracker = start_of_conv - 1;
            }
            else{
                read_kernal();
            }

            ++ram_tracker;

            //if reach the filter's length
            if(count_slide == kernal_len -1){
                ram_tracker = ram_tracker - kernal_len + img_len;
                //cout << "debug_c:" << ram_tracker << " "; //debug
                count_slide = 0;
            }
            else if(!dedicate_to_storing) ++count_slide;

            if(!dedicate_to_storing) ram_addr = ram_tracker;
        }
        //max pooling
        else if(start_pooling && !depth_cnt && !dedicate_to_storing){
            if(total_op_counter==0) debug_register();
            img = ram_data_in;
            //first compare -> set max_value
            if(op_cnt == 0){
                max_pool_value = img;
                ++op_cnt;
                ++ram_tracker;
                ram_addr = ram_tracker;
            }
            //compare
            else if(op_cnt < 3){
                if(max_pool_value<img) max_pool_value = img; 
                ++op_cnt;
                
                //move pointer
                if(op_cnt%2 == 0) ++ram_tracker;
                else ram_tracker = ram_tracker +23;

                ram_addr = ram_tracker;
            }
            //the 4th compare
            //store result_to_ram
            else{
                if(max_pool_value < img) data_store_to_ram(img);
                else data_store_to_ram(max_pool_value);
                op_cnt = 0;

                //move pointer
                ram_tracker = ram_tracker - 23;
            }
            
            if(total_op_counter==3455){
                cout << "pooling is done. (debug)\n";
                start_pooling = false;
                start_conv2 = true;

                //move ram pointer back to start of the pooling layer outputs
                //ram_tracker = begin_of_pool_in_ram;
                begin_of_conv_in_ram = begin_of_pool_in_ram;
                begin_of_pool_in_ram = result_to_ram;

                //debugging
                //debug_register();
                //img_len = 12*12*6;
                //kernal_len = 12;
                //ram_debuging = true;
            }
            else if(total_op_counter%48==47){
                ram_tracker+=24;
            }
            ++total_op_counter;

            //debug
            //if(max_pool_value) cout << max_pool_value << " ";
        }
        //pooling phase 2
        else if(start_pooling2 && !dedicate_to_storing){
            img = ram_data_in;
            //first compare -> set max_value
            if(op_cnt == 0){
                max_pool_value = img;
                ++op_cnt;
                ++ram_tracker;
                ram_addr = ram_tracker;
            }
            //compare
            else if(op_cnt < 3){
                if(max_pool_value<img) max_pool_value = img; 
                ++op_cnt;
                
                //move pointer
                if(op_cnt%2 == 0) ++ram_tracker;
                else ram_tracker = ram_tracker +7;

                ram_addr = ram_tracker;
            }
            //the 4th compare
            //store result_to_ram
            else{
                if(max_pool_value < img) data_store_to_ram(img);
                else data_store_to_ram(max_pool_value);
                op_cnt = 0;

                //move pointer
                ram_tracker = ram_tracker - 7;
            }
            
            if(total_op_counter==1023){
                cout << "pooling 2 is done. (debug)\n";

                start_pooling2 = false;
                signal_fc = true;

                //move ram pointer back to start of the pooling layer outputs
                //ram_tracker = begin_of_pool_in_ram;
                begin_of_pool_in_ram = result_to_ram;

                //debugging
                debug_register();
                //img_len = 256;
                //kernal_len = 4;
                //ram_debuging = true;
            }
            else if(total_op_counter%16==15){
                ram_tracker+=8;
            }
            ++total_op_counter;
        }
        //fc layer1
        else if(start_fclayer1){
            ++total_op_counter;

            if(!dedicate_to_storing){
                img = ram_data_in;
                kernal = rom_data_in;

                if(total_op_counter%257 != 0){
                    sum += img*kernal;
                    rom_addr = rom_tracker++;
                    ram_addr = ram_tracker++;
                }
                //fc1 end
                else if(total_op_counter == 32896){
                    cout << "debug: fc 1 end\n";
                    
                    sum += kernal;
                    sum = relu(sum);
                    data_store_to_ram(sum);
                    start_fclayer1 = false;
                    signal_fc = true;
                    start_pooling2 = true;
                    begin_of_conv_in_ram = begin_of_pool_in_ram;
                    begin_of_pool_in_ram = result_to_ram;

                    //debugging
                    debug_register();
                    //img_len = 128;
                    //kernal_len = 128;
                    //ram_debuging = true;
                }
                else if(total_op_counter%257 == 0){
                    sum += kernal;
                    sum = relu(sum);
                    data_store_to_ram(sum);
                    sum = 0;
                }
            }
            else{
                dedicate_to_storing = false;
                ram_tracker = begin_of_conv_in_ram;
                ram_wr = true; //read mode on
                ram_addr = ram_tracker++;
                rom_addr = rom_tracker++;
                --total_op_counter;
            }
        }
        //fc layer2
        else if(start_fclayer2){
            ++total_op_counter;

            if(!dedicate_to_storing){
                img = ram_data_in;
                kernal = rom_data_in;

                if(total_op_counter%129 != 0){
                    sum += img*kernal;
                    rom_addr = rom_tracker++;
                    ram_addr = ram_tracker++;
                }
                //fc2 end
                else if(total_op_counter == 10836){
                    cout << "debug: fc 2 end\n";
                    
                    sum += kernal;
                    sum = relu(sum);
                    data_store_to_ram(sum);
                    start_fclayer2 = false;
                    start_pooling2 = true;
                    begin_of_conv_in_ram = begin_of_pool_in_ram;
                    begin_of_pool_in_ram = result_to_ram;

                    //debugging
                    debug_register();
                    //img_len = 84;
                    //kernal_len = 84;
                    //ram_debuging = true;
                }
                else if(total_op_counter%129 == 0){
                    sum += kernal;
                    sum = relu(sum);
                    data_store_to_ram(sum);
                    sum = 0;
                }
            }
            else{
                dedicate_to_storing = false;
                ram_tracker = begin_of_conv_in_ram;
                ram_wr = true; //read mode on
                ram_addr = ram_tracker++;
                rom_addr = rom_tracker++;
                --total_op_counter;
            }
        }
        //fc layer3
        else if(start_fclayer3){
            ++total_op_counter;

            if(!dedicate_to_storing){
                img = ram_data_in;
                kernal = rom_data_in;

                if(total_op_counter%85 != 0){
                    sum += img*kernal;
                    rom_addr = rom_tracker++;
                    ram_addr = ram_tracker++;
                }
                //fc2 end
                else if(total_op_counter == 850){
                    cout << "debug: fc 3 end\n";
                    
                    sum += kernal;
                    sum = relu(sum);
                    data_store_to_ram(sum);

                    start_fclayer3 = false;
                    begin_of_conv_in_ram = begin_of_pool_in_ram;
                    begin_of_pool_in_ram = result_to_ram;

                    //debugging
                    debug_register();
                    ram_debuging = true;
                }
                else if(total_op_counter%85 == 0){
                    sum += kernal;
                    sum = relu(sum);
                    data_store_to_ram(sum);
                    sum = 0;
                }
            }
            else{
                dedicate_to_storing = false;
                ram_tracker = begin_of_conv_in_ram;
                ram_wr = true; //read mode on
                ram_addr = ram_tracker++;
                rom_addr = rom_tracker++;
                --total_op_counter;
            }
        }
        else if(dedicate_to_storing){
            dedicate_to_storing = false;
            ram_wr = true; //read mode on

            //check if done 6 times
            if(!start_conv && filter_cnt1 < CONV_LAYER1){
                //move ram pointer back to img (0)
                ram_tracker = 0;
                //cout << "debug---\n";
                request_to_rom = false; //let read_kernal signals conv
                image_read = true;
                start_of_conv = 0;
                //change the start of the filter
                filter_start += 26;
                rom_tracker = filter_start;
                //reset counter
                total_op_counter=0;

                ++filter_cnt1;

                //debug
                debug_register();
            }
            //do convolution layer 2
            else if(start_conv2 && !start_conv && filter_cnt2 < CONV_LAYER2){
                start_of_conv = begin_of_conv_in_ram;
                ram_tracker = start_of_conv;
                ram_addr = ram_tracker;

                img_len = 12;
                request_to_rom = false; //let read_kernal signals conv

                read_kernal();
                
                //reset counter
                total_op_counter=0;
                op_cnt = 0;
                depth_cnt = 1;
                sum = 0;

                ++filter_cnt2;

                cout << "DEBUG conv2: " << filter_cnt2 << endl;
                debug_register();
            }
            //do max pooling
            else if(!start_conv && !start_pooling && !start_conv2 && filter_cnt2 < CONV_LAYER2){
                //rom_tracker+=26; //add up the last kernal
                filter_start=rom_tracker;
                
                start_pooling = true;
                ram_tracker = begin_of_conv_in_ram;
                begin_of_pool_in_ram = result_to_ram;
                ram_addr = ram_tracker;
                total_op_counter = 0;
            }
            //max pooling phase 2
            else if(!start_conv && !start_pooling2 && start_conv2 && filter_cnt2 == CONV_LAYER2){
                //start_pooling = true;
                start_conv2 = false;

                start_pooling2 = true;
                
                op_cnt = 0;

                ram_tracker = begin_of_pool_in_ram;
                begin_of_conv_in_ram = result_to_ram;
                total_op_counter = 0;

                cout << "DEBUG conv2: end" << endl;
                debug_register();
            }
            //fc 1
            else if(!start_pooling2 && signal_fc){
                cout << "debug: enter fc1\n";
                start_fclayer1 = true; 
                signal_fc = false;
                ram_tracker = begin_of_conv_in_ram;
                
                total_op_counter = 0;
                rom_addr = rom_tracker++;
                ram_addr = ram_tracker++;
            }
            //fc 2
            else if(start_pooling2 && !start_fclayer1 && signal_fc){
                cout << "debug: enter fc2\n";
                debug_register();
                ram_tracker = begin_of_conv_in_ram;
                start_fclayer2 = true;
                start_fclayer3 = false;
                start_pooling2 = false;
                signal_fc = false;

                total_op_counter = 0;
                rom_addr = rom_tracker++;
                ram_addr = ram_tracker++;
            }
            //fc 3
            else if(start_pooling2 && !start_fclayer1 && !signal_fc){
                cout << "debug: enter fc3\n";
                debug_register();
                ram_tracker = begin_of_conv_in_ram;
                start_fclayer3 = true;
                start_pooling2 = false;
                start_fclayer1 = false;
                signal_fc = false;

                total_op_counter = 0;
                rom_addr = rom_tracker++;
                ram_addr = ram_tracker++;
            }


            //neccessary at max pooling phase
            if(start_pooling || start_pooling2) ram_addr = ram_tracker;

        }
    }

    //debug check conv result
    if(ram_debuging && !dedicate_to_storing){
        //start = begin_of_conv_in_ram;
        //end = result_to_ram;
        start = 6130;
        end = 6140;
        cut = 24;
        if(start + counting < end){
            ram_wr = 1;
            ram_addr = counting + start;
            //cout << counting + result_to_ram -(img_len);
            img = ram_data_in;
            if(ff++){
                if(img!=0) cout << img << " "; 
                else cout << "0 ";
                counting ++;
            }
            if(counting % cut*cut ==0) cout << endl << endl;
            else if(counting % cut ==0) cout << endl;
        }
        else{
            ram_debuging = false;
            counting = 0;
            ff = 0;
            cout << endl << endl;
            //img_len = 12;
        } 
    }


}

void lenet::data_store_to_ram(TYPE value){
    ram_addr = result_to_ram++;
    ram_data_out = value;
    ram_wr = false;
    dedicate_to_storing = true;
}

TYPE relu(TYPE input){
    if(input < 0) return 0;
    else return input;
}

//do NOT require mannual moving ram_tracker
void lenet::img_write_to_ram(){
    static int first;
    if(first++ == 0){
        //cout << "debug write\n";
        ram_wr = false; //write mode on
    }
    fin >> img;
    img/=255.0;
    ram_addr = ram_tracker++;
    ram_data_out = img;
    if(ram_tracker==IMG_SIZE){
        fin.close();
        ram_wr = true; //close write in to ram
        image_write = false;
        image_read = true;
        result_to_ram = IMG_SIZE + 1; //mark the position we will be using for storing convolution data
        begin_of_conv_in_ram = IMG_SIZE + 1; //mark the starting position
        ram_tracker = 0;
        debug_register();
    }
}

//require mannual moving ram_tracker
void lenet::img_read_from_ram(){
    static int first = 0;
    if(first++ == 0){
        ram_wr = true;
        ram_addr = ram_tracker;
        return;
    }
    if(!dedicate_to_storing) img = ram_data_in;
    //cout << img << " ";
    ram_addr = ram_tracker;
}

//use this function to read kernal value and store into kernal
void lenet::read_kernal(){
    if(request_to_rom){
        //kernal[rom_tracker++] = rom_data_in;
        kernal = rom_data_in;
        start_conv = true; //signal conv to start adding up
        //cout << "Debug: Kernal " << kernal << " ";
        rom_addr = rom_tracker++;
        if( (rom_tracker - filter_start) == KERNAL_SIZE+2 ) rom_tracker = filter_start;
    }
    else{//cout << "debug\n";
        request_to_rom = true; //signal rom to request data
        rom_rd = true;
        rom_addr = filter_start;
        rom_tracker = filter_start + 1;
    }
}