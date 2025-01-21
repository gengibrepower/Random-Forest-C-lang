#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "data_loader.h"

// Função para embaralhar os dados
void shuffle_data(DataPoint *data, int num_samples) {
    for (int i = num_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        DataPoint temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }
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
        for (int i = 0; i < test_size; i++) {
            int prediction = predict_forest(forest, &test_data[i]);
            if (prediction == test_data[i].label) {
                correct_predictions++;
            }
        }

        // Calcular e exibir a precisão
        double accuracy = (double)correct_predictions / test_size * 100.0;
        printf("Accuracy: %.2f%%\n", accuracy);

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