#include<fstream>
#include<random>
#include<time.h>
#include<iostream>
#include<cstring>

using namespace std;
void make_data(int *data_perceptron, int n);

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
        printf("%s\n", "Neural network has this architecture: ");
        for (int i = 0; i < layers; i++){
            printf("%d", size[i]);
            if (i < layers - 1){
                printf("%s", " - ");
            }
        }
        printf("%s", "\n");
        for (int i = 0; i < layers; i++){
            printf("%s", "\n#Layer ");
            printf("%d\n\n", i + 1);
            for(int j = 0; j < size[i]; j++){
                printf("%s", "Neuron #");
                printf("%d", j + 1);
                printf("%s", ": \n");
                printf("%s", "The var of neuron: ");
                printf("%f\n", neurons[i][j].value);
//                if (i < layers - 1){
//                    printf("%s", "Weights: \n");
//                    for (int k = 0; k < size[i + 1]; k++){
//                        printf("%s", "#");
//                        printf("%d", k+1);
//                        printf("%s", ": ");
//                        printf("%f\n", weights[i][j][k]);
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
    const int n = 7;  //количество примеров
    bool to_study = 0;
    int data_n = n;
    int answer_data[data_n];
    int data_perceptron[785*data_n];
    double all_data[data_n][784];
    for (int j = 0; j < data_n; j++) {
        make_data(data_perceptron, data_n);
        for (int i = 0; i <= 785; i++) {
            if (i == 0){
                answer_data[j] = data_perceptron[(j*785)+i];
            }else{
                all_data[j][i] = data_perceptron[(j*785)+i];
            }

        }
    }
    std::cout << "Do I need to study, my Lord? \t Enter 1 or 0 \n";
    std::cin >> to_study;
    double time = 0;

    if (to_study){
        nn.setLayers(l, size);
        for (int e = 0; ra / (n-1) * 100 < 100; e++){
            fout << "Epoch #" << e << endl;
            double epoch_start = clock();
            ra = 0;
            double w_delta = 0;

//            fin.open("train.txt");

            for (int i = 0; i < n-1; i++){
                double start = clock();
                rresult = answer_data[i];
//                fin >> rresult;
                for (int j = 0; j < input_l; j++){
                    input[j] = all_data[i][j] / 256;
//                    fin >> input[j];
//                    printf("%f", input[j]);
                }
                double stop = clock();
                time += stop - start;
                printf("%s", "Number ");
                printf("%d", rresult);
                printf("%s", "\n");
                nn.setInput(input);

                result = nn.forwardFeed();
                //nn.show();
                if (result == rresult){
                    printf("%s", "This bitch is right: ");
                    printf("%d\n", rresult);
                    ra++;
                }
                else{
                    printf("%s", "Result: ");
                    printf("%d", result);
                    printf("%s", " is wrong!\n");
                    printf("%s", "Right result: ");
                    printf("%d", rresult);
                    printf("%s", ", but this bitch has a mistake.\n");
                    nn.backPropogation(result, rresult, 0.6);
                }
            }
//            fin.close();
            double epoch_stop = clock();
            cout << "Right answers: " << ra / (n-1) * 100 << "% /t Max RA: " << double(maxra) / (n-1) * 100 << "(epoch " << maxraepoch << " )" << endl;
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
        for (int i = 0; i < input_l; i++){
            input[i] = all_data[n-1][i] / 256;
        }
        nn.setInput(input);
        result_test = nn.forwardFeed();
        printf("%s", "I think this is ");
        printf("%.0f", result_test);
        printf("%s", "\n\n");
        printf("%s", "And the right result: ");
        printf("%d", answer_data[n-1]);
        printf("%s", "\n\n");
    }
}

void make_data(int *data_perceptron, int n) {
    FILE *file;
    int data_count = 0;
    char sravnenie[4];
    int count = 0;
    char c;

    file = fopen("C:\\Users\\mmmfe\\CLionProjects\\MNIST_CPP\\cmake-build-debug\\train.txt", "r");
    if (file == NULL) {
        printf("Error!\n");
        return;
    }
    while (data_count <= 785 * n) {
        c = getc(file);
        if (c != ',' && c != '\n') {
            sravnenie[count] = c;
            count++;
        } else {
            data_perceptron[data_count] = atoi(sravnenie);
            data_count++;
            count = 0;
            memset(sravnenie, 0, 4);
        }
    }
    fclose(file);
}
