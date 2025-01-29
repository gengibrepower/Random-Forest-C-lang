#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "data_loader.h"

// Felipe Gegembauer


//barra de progresso por pura estética
void print_progress_bar(float progress) {
    int bar_width = 50; // Largura da barra de progresso
    float ratio = (float)progress;
    int pos = bar_width * ratio;

    printf("[");
    for (int i = 0; i < bar_width; i++) {
        if (i < pos) {
            printf("=");
        } else {
            printf(" ");
        }
    }
    printf("] %.2f%%\r", ratio * 100); // \r para retornar ao início da linha
    fflush(stdout); // Força a saída a ser exibida imediatamente
}

// Função para embaralhar os dados
void shuffle_data(DataPoint *data, int num_samples) {
    // Inicializa a semente do gerador de números aleatórios
    srand((unsigned int)time(NULL)); // Inicializa a semente uma vez
    
    for (int i = num_samples - 1; i > 0; i--) {
        // Gera um índice aleatório usando rand()
        int j = rand() % (i + 1);
        
        // Troca os elementos
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
    double predictionerror = (double)(false_positive + false_negative) / test_size;

    // Cálculo da acurácia
    double accuracy = (double)(true_positive + true_negative) / test_size;

    // Cálculo da precisão
    double precision = (true_positive + false_positive > 0) ?                               //toda vez que aparecer o '?' vai ser um if mais reduzido
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
    printf("          P | TP | FN |\n");
    printf(" Real     N | FP | TN |\n");
    printf("              Predicao\n");
    printf("               P   N\n");
    printf("          P | %d | %d |\n", true_negative, false_positive);
    printf("Real      N | %d | %d |\n", false_negative, true_positive);
    printf("Erro de Previsao: %.2f%%\n", predictionerror * 100.0);
    printf("Accuracy: %.2f%%\n",accuracy * 100);
    printf("Precisao: %.2f%%\n", precision * 100.0);
    printf("Recall: %.2f%%\n", recall * 100.0);
    printf("F1 Score: %.2f\n", f1_score);
}

// Função para realizar k-fold cross-validation
void k_fold_cross_validation(DataPoint *data, int num_samples, int num_features, int num_trees, int max_depth, int k_folds) {
    int fold_size = num_samples / k_folds;
    double best_accuracy = 0.0; // Melhor acurácia
    int best_confusion_matrix[2][2] = {0}; // Matriz de confusão da melhor execução

    for (int fold = 0; fold < k_folds; fold++) {
        DataPoint *train_data = malloc((num_samples - fold_size) * sizeof(DataPoint));
        DataPoint *test_data = malloc(fold_size * sizeof(DataPoint));
        int train_index = 0, test_index = 0;

        for (int i = 0; i < num_samples; i++) {
            if (i >= fold * fold_size && i < (fold + 1) * fold_size) {
                test_data[test_index++] = data[i]; // Test fold
            } else {
                train_data[train_index++] = data[i]; // Training fold
            }
        }

        // Chama a funcão para o treinamento
        RandomForest *forest = train_forest(train_data, train_index, num_features, num_trees, max_depth);

        // Fazer previsões no conjunto de teste
        int correct_predictions = 0;
        int confusion_matrix[2][2] = {0}; // Matriz de confusão

        for (int i = 0; i < fold_size; i++) {
            int prediction = predict_forest(forest, &test_data[i]);
            confusion_matrix[test_data[i].label][prediction]++; // Atualiza a matriz de confusão

            if (prediction == test_data[i].label) {
                correct_predictions++;
            }
        }

        // Calcular a acurácia atual
        double current_accuracy = (double)correct_predictions / fold_size;

        // Verifica se a acurácia atual é a melhor
        if (current_accuracy > best_accuracy) {
            best_accuracy = current_accuracy;
            memcpy(best_confusion_matrix, confusion_matrix, sizeof(confusion_matrix)); // Copia a matriz de confusão
        }

        print_progress_bar(0.32 + 0.17*fold); // Atualiza a barra de progresso

        // Liberar memória
        free_forest(forest);
        free(train_data);
        free(test_data);
    }
    print_progress_bar(1.0);
    printf("\n");
    // Exibir a melhor acurácia e a matriz de confusão correspondente
    calculate_metrics(best_confusion_matrix, fold_size); // Exibe as métricas da melhor execução
}

int main() {
    int num_samples, num_features;
    DataPoint *data = load_csv("breast-cancer.csv", &num_samples, &num_features);
    print_progress_bar(0.0);
    if (data) {

        print_progress_bar(0.05);
        // Embaralhar os dados
        srand(time(NULL)); // Inicializa o gerador de números aleatórios
        shuffle_data(data, num_samples);
        print_progress_bar(0.10);
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
        int num_trees = 1000;   // Defina o número de árvores
        int max_depth = 100;    // Profundidade máxima
        int k_folds = 5;        // Número de fold para o cross validation

        print_progress_bar(0.15);

        // Realizar k-fold cross-validation
        k_fold_cross_validation(data, num_samples, num_features, num_trees, max_depth, k_folds);

        // Liberar memória
        for (int i = 0; i < num_samples; i++) {
            free(data[i].features);
        }
        free(data);
    } else {
        printf("Falha ao carregar os dados.\n");
    }

    return 0;
}