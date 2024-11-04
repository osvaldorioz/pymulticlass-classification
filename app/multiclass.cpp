#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <iostream>

//g++ -O2 -Wall -shared -std=c++20 -fPIC `python3.12 -m pybind11 --includes` multiclass.cpp -o multiclass_module`python3.12-config --extension-suffix`

namespace py = pybind11;

// Función para calcular el softmax de un vector
std::vector<double> softmax(const std::vector<double>& logits) {
    double max_logit = *std::max_element(logits.begin(), logits.end());
    double sum_exp = 0.0;
    std::vector<double> exp_values;

    for (double logit : logits) {
        double exp_val = std::exp(logit - max_logit);
        exp_values.push_back(exp_val);
        sum_exp += exp_val;
    }

    for (double& val : exp_values) {
        val /= sum_exp;
    }

    return exp_values;
}

// Clase de clasificación multiclase
class MultiClassClassifier {
public:
    MultiClassClassifier(int num_features, int num_classes)
        : num_features(num_features), num_classes(num_classes) {
        // Inicializar pesos y sesgos con ceros
        weights.resize(num_classes, std::vector<double>(num_features, 0.0));
        biases.resize(num_classes, 0.0);
    }

    // Predicción de la clase usando softmax
    std::vector<double> predict(const std::vector<double>& input) {
        std::vector<double> logits(num_classes, 0.0);

        for (int i = 0; i < num_classes; ++i) {
            double sum = biases[i];
            for (int j = 0; j < num_features; ++j) {
                sum += weights[i][j] * input[j];
            }
            logits[i] = sum;
        }

        return softmax(logits);
    }

    // Entrenamiento usando descenso de gradiente
    void train(const std::vector<std::vector<double>>& inputs, 
               const std::vector<int>& labels, 
               double learning_rate, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                std::vector<double> predictions = predict(inputs[i]);
                for (int j = 0; j < num_classes; ++j) {
                    double error = (j == labels[i] ? 1.0 : 0.0) - predictions[j];
                    for (int k = 0; k < num_features; ++k) {
                        weights[j][k] += learning_rate * error * inputs[i][k];
                    }
                    biases[j] += learning_rate * error;
                }
            }
        }
    }

private:
    int num_features;
    int num_classes;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
};

// Envolvemos la clase con PyBind11
PYBIND11_MODULE(multiclass_module, m) {
    py::class_<MultiClassClassifier>(m, "MultiClassClassifier")
        .def(py::init<int, int>(), py::arg("num_features"), py::arg("num_classes"))
        .def("predict", &MultiClassClassifier::predict)
        .def("train", &MultiClassClassifier::train);
}
