#include <iostream>
#include <fstream>
using namespace std;

int main(){
    ifstream i;
    i.open("lenet_weight_c2.txt");
    double x;
    int k = 0;
    while(i >> x){
        k++;
    }
    cout << k << endl;
    return 0;
}
