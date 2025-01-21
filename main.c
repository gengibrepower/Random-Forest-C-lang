#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "data_loader.h"

// Felipe Gegembauer

// Função para embaralhar os dados
void shuffle_data(DataPoint *data, int num_samples) {
    for (int i = num_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        DataPoint temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }
}

// Função para calcular e exibir a matriz de confusão e as métricas
void calculate_metrics(int confusion_matrix[2][2], int test_size) {
    int true_positive = confusion_matrix[1][1];
    int false_positive = confusion_matrix[0][1];
    int false_negative = confusion_matrix[1][0];
    int true_negative = confusion_matrix[0][0];

    // Cálculo do erro de previsão
    double error = (double)(false_positive + false_negative) / test_size;

    // Cálculo da acurácia
    double accuracy = (double)(true_positive + true_negative) / test_size;

    // Cálculo da precisão
    double precision = (true_positive + false_positive > 0) ? 
                       (double)true_positive / (true_positive + false_positive) : 0.0;

    // Cálculo do recall
    double recall = (true_positive + false_negative > 0) ? 
                    (double)true_positive / (true_positive + false_negative) : 0.0;

    // Cálculo do F1-score
    double f1_score = (precision + recall > 0) ? 
                      (2 * precision * recall) / (precision + recall) : 0.0;

    printf("Matriz de Confusao:\n");
    printf("              Predicao\n");
    printf("               P   N\n");
    printf("          P | %d | %d |\n", true_negative, false_positive);
    printf("Real      N | %d | %d |\n", false_negative, true_positive);
    
    printf("Erro de Previsao: %.2f%%\n", error * 100.0);
    printf("Acuracia: %.2f%%\n", accuracy * 100.0);
    printf("Precisao: %.2f%%\n", precision * 100.0);
    printf("Recall: %.2f%%\n", recall * 100.0);
    printf("F1 Score: %.2f\n", f1_score);
}

int main() {
    int num_samples, num_features;
    DataPoint *data = load_csv("breast-cancer.csv", &num_samples, &num_features);

    if (data) {
        printf("Numero de amostras: %d\n", num_samples);
        printf("Numero de caracteristicas: %d\n", num_features);
        
        // Embaralhar os dados
        srand(time(NULL)); // Inicializa o gerador de números aleatórios
        shuffle_data(data, num_samples);

        // Dividir os dados em 80% para treinamento e 20% para teste
        int train_size = (int)(num_samples * 0.8);
        int test_size = num_samples - train_size;

        DataPoint *train_data = malloc(train_size * sizeof(DataPoint));
        DataPoint *test_data = malloc(test_size * sizeof(DataPoint));

        for (int i = 0; i < train_size; i++) {
            train_data[i] = data[i];
        }
        for (int i = 0; i < test_size; i++) {
            test_data[i] = data[train_size + i];
        }

        // Treinar a floresta aleatória
        int num_trees = 100; // Defina o número de árvores
        int max_depth = 10;  // Defina a profundidade máxima
        RandomForest *forest = train_forest(train_data, train_size, num_features, num_trees, max_depth);

        // Fazer previsões no conjunto de teste
        int correct_predictions = 0;
        int confusion_matrix[2][2] = {0}; // Matriz de confusão para 2 classes

        for (int i = 0; i < test_size; i++) {
            int prediction = predict_forest(forest, &test_data[i]);
            confusion_matrix[test_data[i].label][prediction]++; // Atualiza a matriz de confusão

            if (prediction == test_data[i].label) {
                correct_predictions++;
            }
        }

        // Calcular e exibir a precisão
        double accuracy = (double)correct_predictions / test_size * 100.0;
        printf("Accuracy : %.2f%%\n", accuracy);

        // Calcular e exibir as métricas
        calculate_metrics(confusion_matrix, test_size);

        // Liberar memória
        free_forest(forest);
        free(train_data);
        free(test_data);
        for (int i = 0; i < num_samples; i++) {
            free(data[i].features);
        }
        free(data);
    } else {
        printf("Falha ao carregar os dados.\n");
    }

    return 0;
}