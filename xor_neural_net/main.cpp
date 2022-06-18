#include <iostream>
#include <string>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <fstream>
using namespace std;

float learning_rate = 1.414213562; //  константа для изменения скорости спуска по градиенту
float momentum = 0.5; // насколько сильно влияет предыдущий вес связи

//The weigths for the neural network
float weights[9] = {};

//The training date that will be passed thru the neural network
double training_data[4][2] = { { 1, 0 },
                               { 1, 1 },
                               { 0, 1 },
                               { 0, 0 } };

//The anwsers that the neural network should be targeting for
int anwser_data[4] = { 1, 0, 1, 0 };

int bias = 10;
float h1[16] = {};
float h2[16] = {};
float error[4];
//float output_neuron;
float sum_out[10] = {};
float out_neuron[10] = {};
float gradients[9];
float derivative_O1;
float derivative_h1;
float derivative_h2;
//float sum_output;
float sum_h1[16] = {};
float sum_h2[16] = {};
float update_weights[9];
float prev_weight_update[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
float RMSE_ERROR = 1;
float RMSE_array_error[20000];
float user_input[2];
char choise = 'Y';

////////Prototyping////////
float sigmoid_function(float x);
void calc_hidden_layers(int x);
void calc_output_neuron();
void calc_error(int x);
void calc_derivatives(int x);
void calc_gradient(int x);
void calc_updates();
void update_new_weights();
void calc_RMSE_ERROR();
void generate_weights();
void train_neural_net();
void start_input();
////////Prototyping////////

int main()
{
    generate_weights(); // создам веса при помощи srand() и rand()
    train_neural_net();
    start_input();
    system("pause");
}

void start_input()
{
    while (true)
    {
        if (choise == 'Y' or choise == 'y')
        {
            cout << "enter data 1: "; cin >> user_input[0];
            cout << "enter data 2: "; cin >> user_input[1]; cout << endl;
            sum_h1 = (user_input[0] * weights[0]) + (user_input[1] * weights[2]) + (bias * weights[4]);
            sum_h2 = (user_input[0] * weights[1]) + (user_input[1] * weights[3]) + (bias * weights[5]);
            h1 = sigmoid_function(sum_h1);
            h2 = sigmoid_function(sum_h2);

            sum_output = (h1 * weights[6]) + (h2 * weights[7]) + (bias * weights[8]);
            output_neuron = sigmoid_function(sum_output);
            cout << "result: " << round(output_neuron) << endl;
            cout << "output neuron: " << output_neuron << endl;

            cout << "Again? Press Y";
            cin >> choise;
        }
        else
        {
            break;
        }
    }
}

void train_neural_net()
{
    int epoch = 0;
    while (epoch < 20000)
    {
        for (int i = 0; i < 4; i++)
        {
            calc_hidden_layers(i); // сюда уходит training data
            calc_output_neuron();
            calc_error(i); // использую answer_data
            calc_derivatives(i);
            calc_gradient(i);
            calc_updates();
            update_new_weights();
        }
        cout << "epoch: " << epoch << endl;
        calc_RMSE_ERROR();
        RMSE_array_error[epoch] = RMSE_ERROR;
        epoch = epoch + 1;
    }
}

float sigmoid_function(float x) // дорогая сигмоидда https://miro.medium.com/max/371/1*xqbKX163KWlwvmwPA6bBBw.png
{
    float sigmoid = 1 / (1 + exp(-x));
    return sigmoid;
}

void generate_weights()
{
    srand(time(0)); // чтобы каждый запуск генерировалась разная последовательность
    for (int i = 0; i < 9; i++)
    {
        int randNum = rand() % 2; // rand() даёт целое число
        if (randNum == 1)
            weights[i] = -1 * (double(rand()) / (double(RAND_MAX) + 1.0)); // число от -1.0 до 0.0 не включительно !
        else
            weights[i] = double(rand()) / (double(RAND_MAX) + 1.0); // число от 0.0 до 1.0 не включительно !

        cout << "weight " << i << " = " << weights[i] << endl;
    }
    cout << "" << endl;
}

void calc_hidden_layer_1(int x) // вот такое https://qph.fs.quoracdn.net/main-qimg-ce86322dcbf86b3050d321ec800c45c6
{
    for (int k = 0; k < 16; k++){
        for (int i = 0; i < 784; i++){
            sum_h1[k] += (training_data[x][i] / 256 * weights[784 * k + i]);
        }
    }
    for (int i = 0; i < 16; i++) {
        h1[i] = sigmoid_function(sum_h1[i]);
    }
}

void calc_hidden_layer_2(int x) // вот такое https://qph.fs.quoracdn.net/main-qimg-ce86322dcbf86b3050d321ec800c45c6
{
    int last_weight_number = 784 * 16;
    for (int k = 0; k < 16; k++){
        for (int i = 0; i < 16; i++){
            sum_h2[k] += (h1[i] * weights[last_weight_number + 16 * (k) + i]);
        }
    }
    for (int i = 0; i < 16; i++) {
        h2[i] = sigmoid_function(sum_h2[i]);
    }
}

void calc_output_neurons() // также как и в hidden
{
    int last_weight_number = 784 * 16 + 16 * 16;
    for (int k = 0; k < 10; k++){
        for (int i = 0; i < 16; i++) {
            last_weight_number += (16 * k) + i;
            sum_out[k] += (h2[i] * weights[last_weight_number]);
        }
        last_weight_number += 1;
        sum_out[k] += (bias * weights[last_weight_number]);
    }
    for (int i = 0; i < 10; i++){
        out_neuron[i] = sigmoid_function(sum_out[i]);
    }
}

void calc_error(int x) // значение на выходе - значение для i-ой строчки training_data
{
    error[x] = output_neuron - anwser_data[x];
}

void calc_derivatives(int x) // нужно для градиентого спуска
{
    derivative_O1 = -error[x] * (exp(sum_output) / pow((1 + exp(sum_output)), 2)); // в квадрате
    derivative_h1 = (exp(sum_h1) / pow((1 + exp(sum_h1)), 2)) * weights[6] * derivative_O1;
    derivative_h2 = (exp(sum_h2) / pow((1 + exp(sum_h2)), 2)) * weights[7] * derivative_O1;
}

void calc_gradient(int x) // показываем насколько резский спуск нужно сделать
{
    gradients[0] = sigmoid_function(training_data[x][0]) * derivative_h1;
    gradients[1] = sigmoid_function(training_data[x][0]) * derivative_h2;
    gradients[2] = sigmoid_function(training_data[x][1]) * derivative_h1;
    gradients[3] = sigmoid_function(training_data[x][1]) * derivative_h2;
    gradients[4] = sigmoid_function(bias) * derivative_h1;
    gradients[5] = sigmoid_function(bias) * derivative_h2;
    gradients[6] = h1 * derivative_O1; // от bias
    gradients[7] = h2 * derivative_O1; // от bias
    gradients[8] = sigmoid_function(bias) * derivative_O1; // от bias
}

void calc_updates()
{
    for (int i = 0; i < 9; i++) // насколько нужно изменить вес каждой связи
    {
        update_weights[i] = (learning_rate * gradients[i]) + (momentum * prev_weight_update[i]);
        prev_weight_update[i] = update_weights[i];
    }
}

void update_new_weights()
{
    for (int i = 0; i < 9; i++)
    {
        weights[i] = weights[i] + update_weights[i];
    }
}

void calc_RMSE_ERROR() // самая обычная формула среднеквадратичного отклонения
{
    RMSE_ERROR = sqrt((pow(error[0], 2) + pow(error[1], 2) + pow(error[2], 2) + pow(error[3], 2) / 4));
    cout << "RMSE error: " << RMSE_ERROR << endl;
    cout << "" << endl;
}
