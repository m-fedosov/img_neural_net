#include<fstream>
#include<random>
#include<time.h>
#include<iostream>

using namespace std;

struct neuron {
    double value;
    double error;
    void activate_neuron(){
        value = (1 / (1 + pow(2.71828, -value)));
    }
};

class Neural_Network{
public:
    int layers;
    neuron** neurons;
    double*** weights; //3х мерный массив - 1 разряд = слой нейрона, 2 разряд = номер нейрона, 3 разряд = связь
    int* size;

    double sigmoid_devirative(double x){
        if ((fabs(x - 1) < 1e-9) || (fabs(x) < 1e-9)) return 0.0;
        double result = x * (1.0 - x);
        return result;
    }

    void setLayers(int n, int* p) {
        srand(time(0));
        layers = n;
        neurons = new neuron * [n];
        weights = new double** [n-1];
        size = new int[n];
        for (int i = 0; i < n; i++){
            size[i] = p[i];
            neurons[i] = new neuron[p[i]];
            if (i < n - 1){
                weights[i] = new double* [p[i]];
                for (int j = 0; j < p[i]; j++) {
                    weights[i][j] = new double[p[i + 1]];
                    for (int k = 0; k < p[i + 1]; k++){
                        weights[i][j][k] = (rand() % 100) * 0.01 / size[i];
                    }
                }
            }
        }
    }
    void setLayersNotToStudy(int n, int *p, string filename) {
        ifstream fin;
        fin.open(filename);
        srand(time(0));
        layers = n;
        neurons = new neuron * [n];
        weights = new double** [n - 1];
        size = new int[n];
        for (int i = 0; i < n; i++) {
            size[i] = p[i];
            neurons[i] = new neuron[p[i]];
            if (i < n - 1) {
                weights[i] = new double* [p[i]];
                for (int j = 0; j < p[i]; j++) {
                    weights[i][j] = new double[p[i + 1]];
                    for (int k = 0; k < p[i + 1]; k++) {
                        fin >> weights[i][j][k];
                    }
                }
            }
        }
        fin.open(filename);
    }

    void setInput(double* p){
        for (int i = 0; i < size[0]; i++){
            neurons[0][i].value = p[i];
        }
    }

    void show() {
        cout << "Neural network has this architecture: ";
        for (int i = 0; i < layers; i++){
            cout << size[i];
            if (i < layers - 1){
                cout << " - ";
            }
        }
        cout << endl;
        for (int i = 0; i < layers; i++){
            cout << "\n#Layer " << i+1 << "\n\n";
            for(int j = 0; j < size[i]; j++){
                cout << "Neuron #" << j + 1 << ": \n";
                cout << "The var of neuron: " << neurons[i][j].value << endl;
//                if (i < layers - 1){
//                    cout << "Weights: \n";
//                    for (int k = 0; k < size[i + 1]; k++){
//                        cout << "#" << k+1 << ": ";
//                        cout << weights[i][j][k] << endl;
//                    }
//                }
            }
        }
    }

    void layersCleaner(int LayerNumber, int start, int stop){
        srand(time(0));
        for (int i = start; i < stop; i++){
            neurons[LayerNumber][i].value = 0;
        }
    }

    void forwardFeeder(int LayerNumber, int start, int stop){
        for (int j = start; j < stop; j++){
            for (int k = 0; k < size[LayerNumber - 1]; k++) {
                neurons[LayerNumber][j].value += neurons[LayerNumber - 1][k].value * weights[LayerNumber - 1][k][j];
            }
            neurons[LayerNumber][j].activate_neuron();
        }
    }

    double forwardFeed(){
        for (int i = 1; i < layers; i++){
            layersCleaner(i, 0, size[i]);
            forwardFeeder(i, 0, size[i]);
        }
        double max = 0;
        double prediction = 0;
        for (int i = 0; i < size[layers - 1]; i++){
            if (neurons[layers - 1][i].value > max) {
                max = neurons[layers - 1][i].value;
                prediction = i;
            }
        }
        return prediction;
    }

    void errorCounter(int LayerNumber, int start, int stop, double prediction, double rresult, double lr){
        if (LayerNumber == layers - 1){
            for (int j = start; j < stop; j++){
                if (j != int(rresult)) {
                    neurons[LayerNumber][j].error = -(neurons[LayerNumber][j].value);
                }
                else{
                    neurons[LayerNumber][j].error = 1.0 - neurons[LayerNumber][j].value;
                }
            }
        }
        else{
            for(int j = start; j < stop; j++){
                double error = 0.0;
                for(int k = 0; k < size[LayerNumber + 1]; k++){
                    error += neurons[LayerNumber + 1][k].error * weights[LayerNumber][j][k];
                }
                neurons[LayerNumber][j].error = error;
            }
        }
    }

    void backPropogation(double prediction, double rresult, double lr){
        for (int i = layers - 1; i > 0; i--){
            if (i == layers - 1){
                for (int j = 0; j < size[i]; j++) {
                    if (j != int(rresult)){
                        neurons[i][j].error = -pow((neurons[i][j].value),2);
                    }
                    else {
                        neurons[i][j].error = 1.0 - neurons[i][j].value;
                    }
                }
            }
            else{
                for (int j = 0; j< size[i]; j++){
                    double error = 0.0;
                    for (int k = 0; k < size[i + 1]; k++){
                        error += neurons[i + 1][k].error * weights[i][j][k];
                    }
                    neurons[i][j].error = error;
                }
            }
        }
        double layer_weights_dif = 0;
        for (int i = 0; i < layers - 1; i++){
            layer_weights_dif = 0;
            for (int j = 0; j < size[i]; j++){
                for (int k = 0; k < size[i+1]; k++){
                    weights[i][j][k] += lr * neurons[i + 1][k].error * sigmoid_devirative(neurons[i + 1][k].value) * neurons[i][j].value;
                    double delta = lr * neurons[i + 1][k].error * sigmoid_devirative(neurons[i + 1][k].value) * neurons[i][j].value;
                    layer_weights_dif += lr * neurons[i + 1][k].error * sigmoid_devirative(neurons[i + 1][k].value) * neurons[i][j].value;
                }
            }
            cout << "Layer #" << i << " weights dif: " << layer_weights_dif << endl;
        }
    }

    bool saveWeights(){
        ofstream fout;
        fout.open("weights.txt");
        for (int i = 0; i < layers; i++){
            if (i < layers - 1){
                for (int j = 0; j < size[i]; j++){
                    for (int k = 0;k < size[i+1]; k++){
                        fout << weights[i][j][k] << " ";
                    }
                }
            }
        }
        fout.close();
        return 1;
    }
};

int main(){
    srand(time(0));
    ifstream fin;
    ofstream fout;
    fout.open("log.txt");
    const int l = 4; //количество слоёв
    const int input_l = 784;
    int size[l] = {input_l, 16, 16, 10};
    Neural_Network nn;

    double input[input_l];
    int rresult;
    int result;
    double ra = 0;
    int maxra = 0;
    int maxraepoch = 0;
    const int n = 5;  //количество примеров
    bool to_study = 0;
    double train_data[n][784] = {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,188,255,94,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,191,250,253,93,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,123,248,253,167,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,80,247,253,208,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,207,253,235,77,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,54,209,253,253,88,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,93,254,253,238,170,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,210,254,253,159,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,209,253,254,240,81,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,27,253,253,254,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,206,254,254,198,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,168,253,253,196,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,203,253,248,76,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,188,253,245,93,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,103,253,253,191,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,89,240,253,195,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,220,253,253,80,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,94,253,253,253,94,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,89,251,253,250,131,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,214,218,95,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,30,137,137,192,86,72,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,86,250,254,254,254,254,217,246,151,32,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,179,254,254,254,254,254,254,254,254,254,231,54,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,72,254,254,254,254,254,254,254,254,254,254,254,254,104,0,0,0,0,0,0,0,0,0,0,0,0,0,61,191,254,254,254,254,254,109,83,199,254,254,254,254,243,85,0,0,0,0,0,0,0,0,0,0,0,0,172,254,254,254,202,147,147,45,0,11,29,200,254,254,254,171,0,0,0,0,0,0,0,0,0,0,0,1,174,254,254,89,67,0,0,0,0,0,0,128,252,254,254,212,76,0,0,0,0,0,0,0,0,0,0,47,254,254,254,29,0,0,0,0,0,0,0,0,83,254,254,254,153,0,0,0,0,0,0,0,0,0,0,80,254,254,240,24,0,0,0,0,0,0,0,0,25,240,254,254,153,0,0,0,0,0,0,0,0,0,0,64,254,254,186,7,0,0,0,0,0,0,0,0,0,166,254,254,224,12,0,0,0,0,0,0,0,0,14,232,254,254,254,29,0,0,0,0,0,0,0,0,0,75,254,254,254,17,0,0,0,0,0,0,0,0,18,254,254,254,254,29,0,0,0,0,0,0,0,0,0,48,254,254,254,17,0,0,0,0,0,0,0,0,2,163,254,254,254,29,0,0,0,0,0,0,0,0,0,48,254,254,254,17,0,0,0,0,0,0,0,0,0,94,254,254,254,200,12,0,0,0,0,0,0,0,16,209,254,254,150,1,0,0,0,0,0,0,0,0,0,15,206,254,254,254,202,66,0,0,0,0,0,21,161,254,254,245,31,0,0,0,0,0,0,0,0,0,0,0,60,212,254,254,254,194,48,48,34,41,48,209,254,254,254,171,0,0,0,0,0,0,0,0,0,0,0,0,0,86,243,254,254,254,254,254,233,243,254,254,254,254,254,86,0,0,0,0,0,0,0,0,0,0,0,0,0,0,114,254,254,254,254,254,254,254,254,254,254,239,86,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,182,254,254,254,254,254,254,254,254,243,70,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,76,146,254,255,254,255,146,19,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,141,139,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,254,254,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,254,254,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,254,254,106,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,254,254,184,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,254,254,184,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,254,254,184,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,185,254,184,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,89,254,184,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,146,254,184,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,254,254,184,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,254,254,184,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,254,254,184,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,254,254,184,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,254,254,184,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,156,254,254,184,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,185,255,255,184,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,185,254,254,184,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,185,254,254,184,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,63,254,254,62,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,220,179,6,0,0,0,0,0,0,0,0,9,77,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,28,247,17,0,0,0,0,0,0,0,0,27,202,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,242,155,0,0,0,0,0,0,0,0,27,254,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,160,207,6,0,0,0,0,0,0,0,27,254,65,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,127,254,21,0,0,0,0,0,0,0,20,239,65,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,77,254,21,0,0,0,0,0,0,0,0,195,65,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,70,254,21,0,0,0,0,0,0,0,0,195,142,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,56,251,21,0,0,0,0,0,0,0,0,195,227,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,222,153,5,0,0,0,0,0,0,0,120,240,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,67,251,40,0,0,0,0,0,0,0,94,255,69,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,234,184,0,0,0,0,0,0,0,19,245,69,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,234,169,0,0,0,0,0,0,0,3,199,182,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,154,205,4,0,0,26,72,128,203,208,254,254,131,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,254,129,113,186,245,251,189,75,56,136,254,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,216,233,233,159,104,52,0,0,0,38,254,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,254,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,254,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,206,106,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,186,159,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,209,101,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,25,130,155,254,254,254,157,30,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,103,253,253,253,253,253,253,253,253,114,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,208,253,253,253,253,253,253,253,253,253,253,107,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,253,253,253,253,253,253,253,253,253,253,253,215,101,3,0,0,0,0,0,0,0,0,0,0,0,0,23,210,253,253,253,248,161,222,222,246,253,253,253,253,253,39,0,0,0,0,0,0,0,0,0,0,0,0,136,253,253,253,229,77,0,0,0,70,218,253,253,253,253,215,91,0,0,0,0,0,0,0,0,0,0,5,214,253,253,253,195,0,0,0,0,0,104,224,253,253,253,253,215,29,0,0,0,0,0,0,0,0,0,116,253,253,253,247,75,0,0,0,0,0,0,26,200,253,253,253,253,216,4,0,0,0,0,0,0,0,0,254,253,253,253,195,0,0,0,0,0,0,0,0,26,200,253,253,253,253,5,0,0,0,0,0,0,0,0,254,253,253,253,99,0,0,0,0,0,0,0,0,0,25,231,253,253,253,36,0,0,0,0,0,0,0,0,254,253,253,253,99,0,0,0,0,0,0,0,0,0,0,223,253,253,253,129,0,0,0,0,0,0,0,0,254,253,253,253,99,0,0,0,0,0,0,0,0,0,0,127,253,253,253,129,0,0,0,0,0,0,0,0,254,253,253,253,99,0,0,0,0,0,0,0,0,0,0,139,253,253,253,90,0,0,0,0,0,0,0,0,254,253,253,253,99,0,0,0,0,0,0,0,0,0,78,248,253,253,253,5,0,0,0,0,0,0,0,0,254,253,253,253,216,34,0,0,0,0,0,0,0,33,152,253,253,253,107,1,0,0,0,0,0,0,0,0,206,253,253,253,253,140,0,0,0,0,0,30,139,234,253,253,253,154,2,0,0,0,0,0,0,0,0,0,16,205,253,253,253,250,208,106,106,106,200,237,253,253,253,253,209,22,0,0,0,0,0,0,0,0,0,0,0,82,253,253,253,253,253,253,253,253,253,253,253,253,253,209,22,0,0,0,0,0,0,0,0,0,0,0,0,1,91,253,253,253,253,253,253,253,253,253,253,213,90,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,18,129,208,253,253,253,253,159,129,90,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};
    int answer_data[n] = {1,0,1,4, 0};
    std::cout << "Do I need to study, my Lord? \t Enter 1 or 0 \n";
    std::cin >> to_study;
    double time = 0;

    if (to_study){
        nn.setLayers(l, size);
        for (int e = 0; ra / n * 100 < 100; e++){
            fout << "Epoch #" << e << endl;
            double epoch_start = clock();
            ra = 0;
            double w_delta = 0;

//            fin.open("train.txt");

            for (int i = 0; i < n; i++){
                double start = clock();
                rresult = answer_data[i];
//                fin >> rresult;
                for (int j = 0; j < input_l; j++){
                    input[j] = train_data[i][j] / 256;
//                    fin >> input[j];
//                    cout << input[j] << endl;
                }
                double stop = clock();
                time += stop - start;
                cout << "Number " << rresult << endl;
                nn.setInput(input);

                result = nn.forwardFeed();
                //nn.show();
                if (result == rresult){
                    cout << "This bitch is right: " << rresult << endl;
                    ra++;
                }
                else{
                    cout << "Result: " << result << " is wrong!\n";
                    cout << "Right result: " << rresult << ", but this bitch has a mistake.\n";
                    nn.backPropogation(result, rresult, 0.6);
                }
            }
//            fin.close();
            double epoch_stop = clock();
            cout << "Right answers: " << ra / n * 100 << "% /t Max RA: " << double(maxra) / n * 100 << "(epoch " << maxraepoch << " )" << endl;
            cout << "Time need to fin: " << time / 1000 << " ms\t\t\tEpoch time: " << epoch_stop - epoch_start << endl;
            time = 0;

            if (ra > maxra) {
                maxra = ra;
                maxraepoch = e;
            }
            if (maxraepoch < e - 250){
                maxra = 0;
            }
        }
        if (nn.saveWeights()){
            cout << "Weights saved" << endl;
        }
    }
    else {
        nn.setLayersNotToStudy(l, size, "perfect_weights.txt");
    }

    cout << "Start test? \t Enter 1 or 0 \n";
    bool to_start_test = 0;
    double result_test;
    cin >> to_start_test;
    if(to_start_test){
        fin.open("test.txt");
        double test_data[784] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,220,179,6,0,0,0,0,0,0,0,0,9,77,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,28,247,17,0,0,0,0,0,0,0,0,27,202,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,242,155,0,0,0,0,0,0,0,0,27,254,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,160,207,6,0,0,0,0,0,0,0,27,254,65,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,127,254,21,0,0,0,0,0,0,0,20,239,65,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,77,254,21,0,0,0,0,0,0,0,0,195,65,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,70,254,21,0,0,0,0,0,0,0,0,195,142,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,56,251,21,0,0,0,0,0,0,0,0,195,227,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,222,153,5,0,0,0,0,0,0,0,120,240,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,67,251,40,0,0,0,0,0,0,0,94,255,69,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,234,184,0,0,0,0,0,0,0,19,245,69,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,234,169,0,0,0,0,0,0,0,3,199,182,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,154,205,4,0,0,26,72,128,203,208,254,254,131,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,254,129,113,186,245,251,189,75,56,136,254,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,216,233,233,159,104,52,0,0,0,38,254,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,254,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,254,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,206,106,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,186,159,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,209,101,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        for (int i = 0; i < input_l; i++){
            input[i] = test_data[i] / 256;
        }
        nn.setInput(input);
        result_test = nn.forwardFeed();
        cout << "I think this is " << result_test << "\n\n";
    }
}